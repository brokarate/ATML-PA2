from __future__ import annotations

import argparse
import gc
from collections import Counter
from pathlib import Path

import torch

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json, set_seed
from common.models import load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def _row_id(row: dict, idx: int) -> str:
    for key in ("prompt_id", "source_index", "svamp_id", "problem_id"):
        if key in row:
            return str(row[key])
    return str(idx)


def _outdir(cfg) -> Path:
    return repo_path(cfg["results_dir"]) / "task5_feedback"


def _generation_path(cfg, dataset: str) -> Path:
    return _outdir(cfg) / f"{dataset}_generations.jsonl"


def _summary_path(cfg, dataset: str) -> Path:
    return _outdir(cfg) / f"{dataset}_summary.json"


def _pairwise_path(cfg, dataset: str) -> Path:
    return _outdir(cfg) / f"{dataset}_pairwise.jsonl"


def _load_existing_generations(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    return {(str(row["policy"]), str(row["example_id"])): row for row in rows}


def _generate_policy_rows(cfg, dataset: str, rows: list[dict], policy: str, tokenizer, existing: dict):
    done = {(p, example_id) for (p, example_id) in existing if p == policy}
    missing = [
        (idx, row, _row_id(row, idx))
        for idx, row in enumerate(rows)
        if (policy, _row_id(row, idx)) not in done
    ]
    if not missing:
        print(f"{policy}: found {len(done)} existing {dataset} generations")
        return []

    print(f"{policy}: generating {len(missing)} {dataset} responses")
    model = load_frozen_policy(cfg, policy)
    outputs = []
    max_prompt = int(cfg.get("math_max_prompt_length", 512))
    max_new = int(cfg["math_max_new_tokens"])
    gen_cfg = cfg.get("generation", {})
    for pos, (idx, row, example_id) in enumerate(missing, start=1):
        gen = batch_generate(
            model,
            tokenizer,
            [row["messages"]],
            max_prompt_length=max_prompt,
            max_new_tokens=max_new,
            temperature=float(gen_cfg.get("temperature", 0.7)),
            top_p=float(gen_cfg.get("top_p", 0.9)),
            do_sample=bool(gen_cfg.get("do_sample", True)),
        )
        response = gen["responses"][0]
        pred = extract_designated_final(response)
        reward = exact_reward(response, str(row["gold_final"]))
        outputs.append(
            {
                "dataset": dataset,
                "policy": policy,
                "example_index": idx,
                "example_id": example_id,
                "question": row["question"],
                "gold_final": str(row["gold_final"]),
                "response": response,
                "pred_final": pred,
                "exact_reward": reward,
                "has_designated_final": pred is not None,
                "response_tokens": int(gen["response_lengths"][0]),
                "terminated_with_eos": bool(gen["terminated_with_eos"][0]),
                "truncated": bool(gen["truncated"][0]),
            }
        )
        if pos % 25 == 0:
            print(f"{policy}: {pos}/{len(missing)}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return outputs


def _summarize_generations(rows: list[dict]) -> dict:
    by_policy = {}
    for policy in sorted({row["policy"] for row in rows}):
        sub = [row for row in rows if row["policy"] == policy]
        n = len(sub)
        labels = Counter("correct" if row["exact_reward"] else "wrong" for row in sub)
        by_policy[policy] = {
            "examples": n,
            "exact_accuracy": sum(float(row["exact_reward"]) for row in sub) / n if n else 0.0,
            "designated_final_rate": sum(bool(row["has_designated_final"]) for row in sub) / n if n else 0.0,
            "mean_response_tokens": sum(int(row["response_tokens"]) for row in sub) / n if n else 0.0,
            "terminated_rate": sum(bool(row["terminated_with_eos"]) for row in sub) / n if n else 0.0,
            "truncated_rate": sum(bool(row["truncated"]) for row in sub) / n if n else 0.0,
            "label_distribution": dict(labels),
        }
    return by_policy


def _run_pairwise_judging(cfg, dataset: str, rows: list[dict]) -> list[dict]:
    output_path = _pairwise_path(cfg, dataset)
    existing = read_jsonl(output_path) if output_path.exists() else []
    done = {(row["example_id"], row["pair"]) for row in existing}
    by_example = {}
    for row in rows:
        by_example.setdefault(str(row["example_id"]), {})[row["policy"]] = row

    pairs = [("sft", "rlvr"), ("sft", "rlaif"), ("rlvr", "rlaif")]
    todo = [
        (example_id, pair)
        for example_id, policy_rows in by_example.items()
        for pair in pairs
        if pair[0] in policy_rows and pair[1] in policy_rows and (example_id, f"{pair[0]}_vs_{pair[1]}") not in done
    ]
    if not todo:
        return existing

    judge = PairwiseAIJudge(cfg, _outdir(cfg) / "pairwise_judge_cache.json")
    new_rows = []
    for idx, (example_id, (left, right)) in enumerate(todo, start=1):
        policy_rows = by_example[example_id]
        pref = judge.compare(
            policy_rows[left]["question"],
            policy_rows[left]["response"],
            policy_rows[right]["response"],
        )
        winner = {"A": left, "B": right, "TIE": "tie"}[pref]
        new_rows.append(
            {
                "dataset": dataset,
                "example_id": example_id,
                "pair": f"{left}_vs_{right}",
                "left_policy": left,
                "right_policy": right,
                "judge_preference": pref,
                "winner": winner,
                "left_exact_reward": policy_rows[left]["exact_reward"],
                "right_exact_reward": policy_rows[right]["exact_reward"],
            }
        )
        if idx % 25 == 0:
            write_jsonl(output_path, existing + new_rows)
            print(f"{dataset} pairwise: {idx}/{len(todo)}")
    all_rows = existing + new_rows
    write_jsonl(output_path, all_rows)
    return all_rows


def _summarize_pairwise(rows: list[dict]) -> dict:
    summary = {}
    for pair in sorted({row["pair"] for row in rows}):
        sub = [row for row in rows if row["pair"] == pair]
        n = len(sub)
        wins = Counter(row["winner"] for row in sub)
        exact_agreement = sum(
            1
            for row in sub
            if (row["left_exact_reward"] > row["right_exact_reward"] and row["winner"] == row["left_policy"])
            or (row["right_exact_reward"] > row["left_exact_reward"] and row["winner"] == row["right_policy"])
            or (row["left_exact_reward"] == row["right_exact_reward"] and row["winner"] == "tie")
        )
        summary[pair] = {
            "comparisons": n,
            "winner_counts": dict(wins),
            "judge_exact_agreement_rate": exact_agreement / n if n else 0.0,
            "left_win_rate_ties_half": (wins.get(sub[0]["left_policy"], 0) + 0.5 * wins.get("tie", 0)) / n,
            "right_win_rate_ties_half": (wins.get(sub[0]["right_policy"], 0) + 0.5 * wins.get("tie", 0)) / n,
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    args = ap.parse_args()
    cfg, rows, tokenizer = load_math_evaluation(args.config, args.dataset)
    set_seed(int(cfg.get("seed", 6304)))
    outdir = _outdir(cfg)
    outdir.mkdir(parents=True, exist_ok=True)
    generation_path = _generation_path(cfg, args.dataset)

    existing = _load_existing_generations(generation_path)
    generated = list(existing.values())
    for policy in policy_specs(cfg):
        new_rows = _generate_policy_rows(cfg, args.dataset, rows, policy, tokenizer, existing)
        if new_rows:
            generated.extend(new_rows)
            existing.update({(row["policy"], row["example_id"]): row for row in new_rows})
            generated = sorted(generated, key=lambda r: (r["example_index"], r["policy"]))
            write_jsonl(generation_path, generated)

    generated = read_jsonl(generation_path)
    expected = len(rows) * len(policy_specs(cfg))
    if len(generated) != expected:
        raise RuntimeError(f"Expected {expected} generated rows, found {len(generated)}")

    pairwise = _run_pairwise_judging(cfg, args.dataset, generated)
    summary = {
        "dataset": args.dataset,
        "examples": len(rows),
        "policies": _summarize_generations(generated),
        "pairwise": _summarize_pairwise(pairwise),
        "generations": str(generation_path),
        "pairwise_file": str(_pairwise_path(cfg, args.dataset)),
    }
    save_json(_summary_path(cfg, args.dataset), summary)
    print("Wrote:", generation_path)
    print("Wrote:", _pairwise_path(cfg, args.dataset))
    print("Wrote:", _summary_path(cfg, args.dataset))


if __name__ == "__main__":
    main()
