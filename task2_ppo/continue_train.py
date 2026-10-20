from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.metrics import mean_response_length, sample_entropy, sampled_kl
from common.models import (
    reference_mode,
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards, value_mse_loss


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
        eps=float(cfg.get("adam_epsilon", 1e-5)),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
        eps=float(cfg.get("adam_epsilon", 1e-5)),
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    results_dir = repo_path(cfg["results_dir"]) / run_name
    train_log = results_dir / "train_log.jsonl"
    if train_log.exists():
        train_log.unlink()

    policy = bundle["policy"]
    value_model = bundle["value_model"]
    tokenizer = bundle["tokenizer"]
    reward_model = bundle["reward_model"]
    reward_tokenizer = bundle["reward_tokenizer"]
    prompt_rows = bundle["prompt_rows"]
    policy_optimizer = bundle["policy_optimizer"]
    value_optimizer = bundle["value_optimizer"]
    timer = wall_timer()

    device = next(policy.parameters()).device
    total_updates = int(cfg["updates"])
    prompts_per_update = int(cfg["prompts_per_update"])
    ppo_epochs = int(cfg["ppo_epochs"])
    max_response_length = int(cfg["max_response_length"])
    max_prompt_length = int(cfg["max_prompt_length"])
    max_grad_norm = float(cfg["max_grad_norm"])

    peak_memory = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    progress = tqdm(range(total_updates), desc=f"{run_name} PPO")
    for update_idx in progress:
        policy_grad_norm = torch.tensor(0.0, device=device)
        value_grad_norm = torch.tensor(0.0, device=device)
        skipped_update = False
        start = (update_idx * prompts_per_update) % len(prompt_rows)
        selected = [prompt_rows[(start + offset) % len(prompt_rows)] for offset in range(prompts_per_update)]
        prompts = [prompt_messages(row) for row in selected]

        gen = batch_generate(
            policy,
            tokenizer,
            prompts,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_response_length,
            temperature=float(cfg["generation"]["temperature"]),
            top_p=float(cfg["generation"]["top_p"]),
            do_sample=bool(cfg["generation"]["do_sample"]),
        )
        sequences = gen["sequences"].detach().clone()
        attention_mask = gen["attention_mask"].detach().clone()
        response_ids = gen["response_ids"].detach().clone()
        response_mask = gen["response_mask"].detach().clone()
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
            all_values = token_values(value_model, sequences.to(next(value_model.parameters()).device), attention_mask.to(next(value_model.parameters()).device))
            old_values = all_values[:, prompt_width - 1 : prompt_width - 1 + response_ids.shape[1]].to(device)
            task_reward = score_reward_pairs(
                reward_model,
                reward_tokenizer,
                prompts,
                gen["responses"],
                max_length=int(cfg["reward_max_length"]),
            ).to(device)
            missing_eos = torch.tensor(
                [0.0 if done else float(cfg["missing_eos_penalty"]) for done in gen["terminated_with_eos"]],
                device=device,
            )
            effective_reward = task_reward - missing_eos
            rewards = shaped_rewards(effective_reward, old_logp, ref_logp, response_mask, float(cfg["kl_beta"]))
            advantages, returns = compute_gae(
                rewards,
                old_values,
                response_mask,
                gamma=float(cfg["gamma"]),
                lam=float(cfg["gae_lambda"]),
            )
            advantages = normalize_advantages(advantages, response_mask).detach()
            returns = returns.detach()

        last_policy_loss = last_value_loss = last_clip_fraction = last_entropy = torch.tensor(0.0)
        for _ in range(ppo_epochs):
            policy_optimizer.zero_grad(set_to_none=True)
            value_optimizer.zero_grad(set_to_none=True)

            new_logp, _ = response_token_logprobs(
                policy,
                sequences,
                attention_mask,
                prompt_width,
                response_ids,
            )
            values = token_values(
                value_model,
                sequences.to(next(value_model.parameters()).device),
                attention_mask.to(next(value_model.parameters()).device),
            )
            values = values[:, prompt_width - 1 : prompt_width - 1 + response_ids.shape[1]].to(device)

            policy_loss, _, clip_fraction = ppo_policy_loss(
                new_logp,
                old_logp,
                advantages,
                response_mask,
                eps=float(cfg["clip_epsilon"]),
            )
            value_loss = value_mse_loss(values, returns, response_mask)
            loss = policy_loss + float(cfg["value_coef"]) * value_loss
            if not torch.isfinite(loss):
                skipped_update = True
                policy_optimizer.zero_grad(set_to_none=True)
                value_optimizer.zero_grad(set_to_none=True)
                break
            loss.backward()

            policy_grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_grad_norm)
            value_grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(value_model), max_grad_norm)
            if not torch.isfinite(policy_grad_norm) or not torch.isfinite(value_grad_norm):
                skipped_update = True
                policy_optimizer.zero_grad(set_to_none=True)
                value_optimizer.zero_grad(set_to_none=True)
                break
            policy_optimizer.step()
            value_optimizer.step()

            last_policy_loss = policy_loss.detach()
            last_value_loss = value_loss.detach()
            last_clip_fraction = clip_fraction.detach()
            last_entropy = sample_entropy(new_logp.detach(), response_mask).detach()

        if torch.cuda.is_available():
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated())

        metrics = {
            "run": run_name,
            "update": update_idx + 1,
            "clip_epsilon": float(cfg["clip_epsilon"]),
            "kl_beta": float(cfg["kl_beta"]),
            "learned_reward": float(task_reward.mean().detach().cpu()),
            "effective_reward": float(effective_reward.mean().detach().cpu()),
            "kl": float(sampled_kl(old_logp.detach(), ref_logp.detach(), response_mask).cpu()),
            "policy_loss": float(last_policy_loss.cpu()),
            "value_loss": float(last_value_loss.cpu()),
            "entropy": float(last_entropy.cpu()),
            "clip_fraction": float(last_clip_fraction.cpu()),
            "policy_grad_norm": float(policy_grad_norm.detach().cpu()),
            "value_grad_norm": float(value_grad_norm.detach().cpu()),
            "response_length": mean_response_length(response_mask),
            "terminated_rate": float(sum(gen["terminated_with_eos"]) / len(gen["terminated_with_eos"])),
            "skipped_update": bool(skipped_update),
            "wall_time_seconds": timer(),
        }
        append_jsonl(train_log, metrics)
        progress.set_postfix(
            reward=metrics["learned_reward"],
            kl=metrics["kl"],
            clip=metrics["clip_fraction"],
        )

    policy.save_pretrained(out)
    tokenizer.save_pretrained(out)
    value_out = out / "value_model"
    value_model.save_pretrained(value_out)
    summary = {
        "run": run_name,
        "updates": total_updates,
        "output_dir": str(out.relative_to(repo_path(".")) if out.is_relative_to(repo_path(".")) else out),
        "value_output_dir": str(value_out.relative_to(repo_path(".")) if value_out.is_relative_to(repo_path(".")) else value_out),
        "clip_epsilon": float(cfg["clip_epsilon"]),
        "kl_beta": float(cfg["kl_beta"]),
        "wall_time_seconds": timer(),
        "peak_vram_bytes": int(peak_memory),
    }
    save_json(results_dir / "summary.json", summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
