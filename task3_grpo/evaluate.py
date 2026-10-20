from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json
from common.metrics import sample_entropy, sampled_kl
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    result = evaluate_policy(args.config, args.adapter, args.name)
    print(result)


@torch.no_grad()
def evaluate_policy(config_path: str, adapter: str, name: str):
    bundle = load_evaluation_bundle(config_path, adapter)
    cfg = bundle["cfg"]
    rows = bundle["rows"]
    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]
    reward_model, reward_tokenizer = bundle["reward"]

    gen_batch_size = int(cfg.get("eval_batch_size", cfg["num_generations"]))
    prompts = [prompt_messages(row) for row in rows]
    generations = []
    rewards = []
    kls = []
    entropies = []
    lengths = []

    for start in tqdm(range(0, len(prompts), gen_batch_size), desc=f"{name} eval"):
        prompt_batch = prompts[start : start + gen_batch_size]
        row_batch = rows[start : start + gen_batch_size]
        gen = batch_generate(
            policy,
            tokenizer,
            prompt_batch,
            max_prompt_length=int(cfg["max_prompt_length"]),
            max_new_tokens=int(cfg["max_completion_length"]),
            temperature=float(cfg["generation"]["temperature"]),
            top_p=float(cfg["generation"]["top_p"]),
            do_sample=bool(cfg["generation"]["do_sample"]),
        )
        policy_logp, _ = response_token_logprobs(
            policy,
            gen["sequences"],
            gen["attention_mask"],
            gen["prompt_width"],
            gen["response_ids"],
        )
        with reference_mode(policy):
            ref_logp, _ = response_token_logprobs(
                policy,
                gen["sequences"],
                gen["attention_mask"],
                gen["prompt_width"],
                gen["response_ids"],
            )
        scores = score_reward_pairs(
            reward_model,
            reward_tokenizer,
            prompt_batch,
            gen["responses"],
            max_length=int(cfg.get("reward_max_length", 1024)),
        )
        rewards.extend(float(x) for x in scores.cpu())
        kls.append(float(sampled_kl(policy_logp, ref_logp, gen["response_mask"]).cpu()))
        entropies.append(float(sample_entropy(policy_logp, gen["response_mask"]).cpu()))
        lengths.extend(float(x) for x in gen["response_lengths"])
        for row, response, length, terminated, truncated in zip(
            row_batch,
            gen["responses"],
            gen["response_lengths"],
            gen["terminated_with_eos"],
            gen["truncated"],
        ):
            generations.append(
                {
                    "prompt_id": row.get("prompt_id"),
                    "source_index": row.get("source_index"),
                    "response": response,
                    "response_length": int(length),
                    "terminated_with_eos": bool(terminated),
                    "truncated": bool(truncated),
                }
            )

    result = {
        "name": name,
        "adapter": adapter,
        "examples": len(generations),
        "reward_model_score": sum(rewards) / max(len(rewards), 1),
        "sampled_kl": sum(kls) / max(len(kls), 1),
        "entropy": sum(entropies) / max(len(entropies), 1),
        "mean_response_length": sum(lengths) / max(len(lengths), 1),
        "terminated_rate": sum(1.0 for row in generations if row["terminated_with_eos"]) / max(len(generations), 1),
        "truncated_rate": sum(1.0 for row in generations if row["truncated"]) / max(len(generations), 1),
    }
    result_dir = repo_path(cfg["results_dir"]) / name
    save_json(result_dir / "eval_summary.json", result)
    write_jsonl(result_dir / "generations.jsonl", generations)
    return result


if __name__ == "__main__":
    main()
