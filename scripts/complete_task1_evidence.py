"""Frozen-checkpoint diagnostics only; never trains or changes existing results."""
from __future__ import annotations

import gc
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

from common.data import (encode_prompt_response, load_yaml, pad_batch,
                         preference_responses, prompt_messages_from_preference,
                         read_jsonl, repo_path, write_jsonl)
from common.generation import batch_generate, response_sequence_logprobs
from common.logging_utils import append_jsonl, save_json, set_seed
from common.metrics import word_count, parse_word_limit
from common.models import load_policy, load_tokenizer, reference_mode
from task1_dpo.train import filter_overlong_prompts


def main():
    torch.set_num_threads(4)
    cfg = load_yaml("configs/dpo.yaml")
    tokenizer = load_tokenizer(cfg["base_model"])
    rows, skipped = filter_overlong_prompts(
        tokenizer, read_jsonl(cfg["paths"]["dpo_length_eval"]),
        int(cfg["max_sequence_length"]))
    out = repo_path("results/task1_dpo/completion_diagnostics")
    out.mkdir(parents=True, exist_ok=True)
    ref_path = out / "reference_logprobs.jsonl"
    reference = {r["prompt_id"]: r for r in read_jsonl(ref_path)} if ref_path.exists() else {}
    summary = {"provenance": "Frozen post-hoc evaluation of existing checkpoints; no training.",
               "skipped_overlong_prompts": len(skipped), "runs": {}}
    for run, adapter in [("standard", cfg["standard_output"]),
                         ("length_balanced", cfg["length_output"])]:
        model = load_policy(cfg, adapter_path=adapter, trainable=False)
        device = next(model.parameters()).device
        path = out / f"{run}_preference_examples.jsonl"
        done = {r["prompt_id"]: r for r in read_jsonl(path)} if path.exists() else {}
        for i, row in enumerate(rows):
            pid = row["prompt_id"]
            if pid in done:
                continue
            prompt = prompt_messages_from_preference(row)
            responses = preference_responses(row)
            batch = pad_batch(tokenizer, [encode_prompt_response(
                tokenizer, prompt, response, int(cfg["max_sequence_length"]))
                for response in responses])
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                policy, _, _ = response_sequence_logprobs(model, batch)
                if pid not in reference:
                    with reference_mode(model):
                        ref, _, _ = response_sequence_logprobs(model, batch)
                    r = {"prompt_id": pid, "chosen_logp": float(ref[0]),
                         "rejected_logp": float(ref[1])}
                    append_jsonl(ref_path, r)
                    reference[pid] = r
            r = reference[pid]
            margin = float(policy[0] - policy[1]) - (r["chosen_logp"] - r["rejected_logp"])
            item = {"prompt_id": pid, "source_index": row["source_index"],
                    "length_stratum": row["length_stratum"],
                    "policy_chosen_logp": float(policy[0]),
                    "policy_rejected_logp": float(policy[1]),
                    "reference_adjusted_margin": margin,
                    "preferred_correctly": margin > 0}
            append_jsonl(path, item)
            done[pid] = item
            if (i + 1) % 20 == 0:
                print(f"{run}: preference {i + 1}/{len(rows)}", flush=True)
        strata = {}
        for stratum in sorted({r["length_stratum"] for r in rows}):
            sub = [done[r["prompt_id"]] for r in rows if r["length_stratum"] == stratum]
            strata[stratum] = {"examples": len(sub), "preference_accuracy":
                               sum(r["preferred_correctly"] for r in sub) / len(sub)}
        # Reuse the original generations for dispersion; do not regenerate them.
        source = ("standard_length_stratified" if run == "standard" else "length_balanced_stratified")
        lengths = torch.tensor([r["response_length"] for r in read_jsonl(
            f"results/task1_dpo/{source}/generations.jsonl")], dtype=torch.float64)
        word_path = out / f"{run}_word_limit_generations.jsonl"
        words = {r["prompt_id"]: r for r in read_jsonl(word_path)} if word_path.exists() else {}
        set_seed(int(cfg["seed"]))
        for row in read_jsonl(cfg["paths"]["word_limit_prompts"]):
            if row["prompt_id"] in words:
                continue
            with torch.no_grad():
                gen = batch_generate(model, tokenizer, [row["messages"]],
                    max_prompt_length=int(cfg["max_sequence_length"]),
                    max_new_tokens=int(cfg["max_generation_tokens"]),
                    **cfg["generation"])
            prompt = row["messages"][-1]["content"]
            count = word_count(gen["responses"][0])
            limit = parse_word_limit(prompt)
            # 'under N' is strict; 'at most/no more than N' is inclusive.
            strict = f"under {limit} words" in prompt.lower()
            item = {"prompt_id": row["prompt_id"], "prompt": prompt,
                    "response": gen["responses"][0], "response_tokens": gen["response_lengths"][0],
                    "word_count": count, "word_limit": limit,
                    "strict_under": strict, "compliant": count < limit if strict else count <= limit}
            append_jsonl(word_path, item)
            words[row["prompt_id"]] = item
            print(f"{run}: word-limit {len(words)}/10", flush=True)
        summary["runs"][run] = {"strata": strata,
            "aggregate_preference_accuracy": sum(done[r["prompt_id"]]["preferred_correctly"] for r in rows)/len(rows),
            "saved_generation_mean_tokens": float(lengths.mean()),
            "saved_generation_sample_sd_tokens": float(lengths.std()),
            "word_limit_examples": len(words),
            "word_limit_compliance": sum(r["compliant"] for r in words.values())/len(words),
            "word_count_definition": "common.metrics.word_count: Unicode word-boundary regex",
            "stratum_source": cfg["paths"]["dpo_length_eval"]}
        save_json(out / "summary.json", summary)
        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    print(summary, flush=True)


if __name__ == "__main__":
    main()
