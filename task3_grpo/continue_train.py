from __future__ import annotations

import argparse

import torch
from torch.optim import AdamW
from tqdm import tqdm

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.metrics import mean_response_length
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode, trainable_parameters
from task3_grpo.grpo import (
    group_relative_advantages,
    grpo_policy_loss,
    mask_truncated_sequences,
)


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["learning_rate"]),
        eps=float(cfg.get("adam_epsilon", 1e-5)),
    )
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo", run_name: str = "standard"):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    policy = bundle["policy"]
    tokenizer = bundle["tokenizer"]
    reward_model = bundle["reward_model"]
    reward_tokenizer = bundle["reward_tokenizer"]
    prompt_rows = bundle["prompt_rows"]
    optimizer = bundle["optimizer"]
    device = next(policy.parameters()).device

    results_dir = repo_path(cfg["results_dir"]) / run_name
    train_log = results_dir / "train_log.jsonl"
    if train_log.exists():
        train_log.unlink()

    total_updates = int(cfg["updates"])
    prompts_per_update = int(cfg["prompts_per_update"])
    k = int(cfg["num_generations"])
    policy_epochs = int(cfg["policy_epochs"])
    max_grad_norm = float(cfg["max_grad_norm"])
    timer = wall_timer()
    peak_memory = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    progress = tqdm(range(total_updates), desc=f"{run_name} GRPO")
    for update_idx in progress:
        grad_norm = torch.tensor(0.0, device=device)
        skipped_update = False
        start = (update_idx * prompts_per_update) % len(prompt_rows)
        selected = [prompt_rows[(start + offset) % len(prompt_rows)] for offset in range(prompts_per_update)]
        base_prompts = [prompt_messages(row) for row in selected]
        prompts = [prompt for prompt in base_prompts for _ in range(k)]
        group_ids = torch.arange(prompts_per_update, device=device).repeat_interleave(k)

        gen = batch_generate(
            policy,
            tokenizer,
            prompts,
            max_prompt_length=int(cfg["max_prompt_length"]),
            max_new_tokens=int(cfg["max_completion_length"]),
            temperature=float(cfg["generation"]["temperature"]),
            top_p=float(cfg["generation"]["top_p"]),
            do_sample=bool(cfg["generation"]["do_sample"]),
        )
        sequences = gen["sequences"].detach().clone()
        attention_mask = gen["attention_mask"].detach().clone()
        response_ids = gen["response_ids"].detach().clone()
        token_mask = gen["response_mask"].detach().clone()
        if bool(cfg.get("mask_truncated_completions", False)):
            token_mask = mask_truncated_sequences(token_mask, gen["truncated"])
        prompt_width = int(gen["prompt_width"])

        with torch.no_grad():
            old_logp, _ = response_token_logprobs(
                policy,
                sequences,
                attention_mask,
                prompt_width,
                response_ids,
            )
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(
                    policy,
                    sequences,
                    attention_mask,
                    prompt_width,
                    response_ids,
                )
            rewards = score_reward_pairs(
                reward_model,
                reward_tokenizer,
                prompts,
                gen["responses"],
                max_length=int(cfg.get("reward_max_length", 1024)),
            ).to(device)
            seq_adv = group_relative_advantages(rewards, group_ids)
            group_stds = []
            for gid in torch.unique(group_ids):
                group_stds.append(float(rewards[group_ids == gid].std(unbiased=False).detach().cpu()))
            uninformative = sum(1 for value in group_stds if value <= 1e-6) / max(len(group_stds), 1)

        last_loss = last_policy_term = last_kl = last_clip = last_entropy = torch.tensor(0.0, device=device)
        for _ in range(policy_epochs):
            optimizer.zero_grad(set_to_none=True)
            new_logp, _ = response_token_logprobs(
                policy,
                sequences,
                attention_mask,
                prompt_width,
                response_ids,
            )
            loss, diagnostics = grpo_policy_loss(
                new_logp,
                old_logp,
                seq_adv,
                token_mask,
                ref_logp,
                eps=float(cfg["clip_epsilon"]),
                beta=float(cfg["kl_beta"]),
                loss_type=loss_type,
                max_completion_length=int(cfg["max_completion_length"]),
            )
            if not torch.isfinite(loss):
                skipped_update = True
                optimizer.zero_grad(set_to_none=True)
                break
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_grad_norm)
            if not torch.isfinite(grad_norm):
                skipped_update = True
                optimizer.zero_grad(set_to_none=True)
                break
            optimizer.step()

            last_loss = loss.detach()
            last_policy_term = diagnostics["policy_term"].detach()
            last_kl = diagnostics["sampled_kl"].detach()
            last_clip = diagnostics["clip_fraction"].detach()
            last_entropy = diagnostics["sample_entropy"].detach()

        if torch.cuda.is_available():
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated())

        metrics = {
            "run": run_name,
            "update": update_idx + 1,
            "loss_type": loss_type,
            "reward": float(rewards.mean().detach().cpu()),
            "group_reward_std": float(sum(group_stds) / max(len(group_stds), 1)),
            "uninformative_group_fraction": float(uninformative),
            "loss": float(last_loss.cpu()),
            "policy_term": float(last_policy_term.cpu()),
            "kl": float(last_kl.cpu()),
            "clip_fraction": float(last_clip.cpu()),
            "entropy": float(last_entropy.cpu()),
            "grad_norm": float(grad_norm.detach().cpu()),
            "response_length": mean_response_length(gen["response_mask"]),
            "masked_response_length": mean_response_length(token_mask),
            "terminated_rate": float(sum(gen["terminated_with_eos"]) / len(gen["terminated_with_eos"])),
            "truncated_rate": float(sum(gen["truncated"]) / len(gen["truncated"])),
            "skipped_update": bool(skipped_update),
            "wall_time_seconds": timer(),
        }
        append_jsonl(train_log, metrics)
        progress.set_postfix(reward=metrics["reward"], kl=metrics["kl"], std=metrics["group_reward_std"])

    policy.save_pretrained(out)
    tokenizer.save_pretrained(out)
    summary = {
        "run": run_name,
        "updates": total_updates,
        "loss_type": loss_type,
        "output_dir": str(out.relative_to(repo_path(".")) if out.is_relative_to(repo_path(".")) else out),
        "wall_time_seconds": timer(),
        "peak_vram_bytes": int(peak_memory),
    }
    save_json(results_dir / "summary.json", summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()
