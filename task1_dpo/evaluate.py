from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.data import (
    load_yaml,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
    write_jsonl,
)
from common.generation import (
    batch_generate,
    response_sequence_logprobs,
    response_token_logprobs,
    score_reward_pairs,
)
from common.logging_utils import save_json
from common.metrics import sampled_kl
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from task1_dpo.dpo import dpo_loss
from task1_dpo.train import filter_overlong_prompts, make_collate


def load_evaluation_bundle(config_path: str, adapter: str, eval_path: str | None = None):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(eval_path or cfg["paths"]["dpo_standard_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
    }


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


@torch.no_grad()
def evaluate_adapter(
    config_path: str,
    adapter: str,
    name: str,
    eval_path: str | None = None,
    max_examples: int | None = None,
) -> dict:
    bundle = load_evaluation_bundle(config_path, adapter, eval_path)
    cfg = bundle["cfg"]
    rows = bundle["rows"][:max_examples] if max_examples else bundle["rows"]
    tokenizer = bundle["tokenizer"]
    rows, skipped = filter_overlong_prompts(tokenizer, rows, int(cfg["max_sequence_length"]))
    if skipped:
        print(f"Skipped overlong eval prompts: {len(skipped)}")
    policy = bundle["policy"]
    beta = float(cfg["beta"])
    device = next(policy.parameters()).device

    pair_loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )

    totals = {
        "loss": 0.0,
        "preference_accuracy": 0.0,
        "logit_mean": 0.0,
        "policy_margin_mean": 0.0,
    }
    count = 0
    for chosen, rejected in tqdm(pair_loader, desc=f"{name} preference eval", leave=False):
        chosen = {k: v.to(device) for k, v in chosen.items()}
        rejected = {k: v.to(device) for k, v in rejected.items()}
        policy_chosen, _, _ = response_sequence_logprobs(policy, chosen)
        policy_rejected, _, _ = response_sequence_logprobs(policy, rejected)
        with reference_mode(policy):
            ref_chosen, _, _ = response_sequence_logprobs(policy, chosen)
            ref_rejected, _, _ = response_sequence_logprobs(policy, rejected)
        loss, diagnostics = dpo_loss(
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
            beta,
        )
        batch_n = int(chosen["input_ids"].shape[0])
        count += batch_n
        totals["loss"] += float(loss.cpu()) * batch_n
        for key in ("preference_accuracy", "logit_mean", "policy_margin_mean"):
            totals[key] += float(diagnostics[key].cpu()) * batch_n

    preference_summary = {key: value / max(count, 1) for key, value in totals.items()}
    preference_summary["examples"] = count

    prompts = [prompt_messages_from_preference(row) for row in rows]
    generations = []
    kl_values = []
    response_lengths = []
    gen_batch_size = int(cfg.get("generation_batch_size", cfg["batch_size"]))
    for start in tqdm(range(0, len(prompts), gen_batch_size), desc=f"{name} generate", leave=False):
        prompt_batch = prompts[start : start + gen_batch_size]
        gen = batch_generate(
            policy,
            tokenizer,
            prompt_batch,
            max_prompt_length=int(cfg["max_sequence_length"]),
            max_new_tokens=int(cfg["max_generation_tokens"]),
            temperature=float(cfg["generation"]["temperature"]),
            top_p=float(cfg["generation"]["top_p"]),
            do_sample=bool(cfg["generation"]["do_sample"]),
        )
        policy_tok_logp, _ = response_token_logprobs(
            policy,
            gen["sequences"],
            gen["attention_mask"],
            gen["prompt_width"],
            gen["response_ids"],
        )
        with reference_mode(policy):
            ref_tok_logp, _ = response_token_logprobs(
                policy,
                gen["sequences"],
                gen["attention_mask"],
                gen["prompt_width"],
                gen["response_ids"],
            )
        kl_values.append(float(sampled_kl(policy_tok_logp, ref_tok_logp, gen["response_mask"]).cpu()))
        response_lengths.extend(float(x) for x in gen["response_lengths"])
        for row, response, length in zip(rows[start : start + gen_batch_size], gen["responses"], gen["response_lengths"]):
            generations.append(
                {
                    "id": row.get("id", row.get("prompt_id")),
                    "response": response,
                    "response_length": int(length),
                }
            )

    reward_scores = []
    reward_model, reward_tokenizer = load_reward_model(cfg)
    for start in tqdm(range(0, len(generations), gen_batch_size), desc=f"{name} reward", leave=False):
        prompt_batch = prompts[start : start + gen_batch_size]
        response_batch = [row["response"] for row in generations[start : start + gen_batch_size]]
        scores = score_reward_pairs(
            reward_model,
            reward_tokenizer,
            prompt_batch,
            response_batch,
            max_length=int(cfg.get("reward_max_length", 1024)),
        )
        reward_scores.extend(float(x) for x in scores.cpu())

    result = {
        "name": name,
        "adapter": adapter,
        "eval_file": eval_path or cfg["paths"]["dpo_standard_eval"],
        "skipped_overlong_prompts": len(skipped),
        "preference": preference_summary,
        "generation": {
            "examples": len(generations),
            "sampled_kl": _mean(kl_values),
            "mean_response_length": _mean(response_lengths),
            "reward_model_score": _mean(reward_scores),
        },
    }
    result_dir = repo_path(cfg["results_dir"]) / name
    save_json(result_dir / "eval_summary.json", result)
    write_jsonl(result_dir / "generations.jsonl", generations)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--eval-path")
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    evaluate_adapter(args.config, args.adapter, args.name, args.eval_path, args.max_examples)


if __name__ == "__main__":
    main()
