from __future__ import annotations

import argparse
import torch

from common.data import load_yaml, repo_path
from common.logging_utils import save_json
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import evaluate_policy
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def _pad_1d(tensors, value=0.0):
    width = max(int(x.numel()) for x in tensors)
    out = torch.full((len(tensors), width), float(value), dtype=torch.float32)
    mask = torch.zeros((len(tensors), width), dtype=torch.float32)
    for idx, tensor in enumerate(tensors):
        tensor = tensor.float().flatten()
        out[idx, : tensor.numel()] = tensor
        mask[idx, : tensor.numel()] = 1.0
    return out, mask


def cached_clipping_diagnostics(rows, cfg):
    old_logp, response_mask = _pad_1d([row["old_logprobs"] for row in rows])
    ref_logp, _ = _pad_1d([row["ref_logprobs"] for row in rows])
    values, _ = _pad_1d([row["values"] for row in rows])
    reward = torch.tensor([float(row["effective_terminal_reward"]) for row in rows], dtype=torch.float32)

    rewards = shaped_rewards(reward, old_logp, ref_logp, response_mask, float(cfg["kl_beta"]))
    advantages, _ = compute_gae(
        rewards,
        values,
        response_mask,
        gamma=float(cfg["gamma"]),
        lam=float(cfg["gae_lambda"]),
    )
    advantages = normalize_advantages(advantages, response_mask)

    diagnostics = []
    for eps in cfg["clip_values"]:
        loss, ratio, clip_fraction = ppo_policy_loss(
            ref_logp,
            old_logp,
            advantages,
            response_mask,
            eps=float(eps),
        )
        affected = ((ratio < 1.0 - float(eps)) | (ratio > 1.0 + float(eps))).float()
        diagnostics.append(
            {
                "clip_epsilon": float(eps),
                "cached_surrogate_loss": float(loss.cpu()),
                "cached_clip_fraction": float(clip_fraction.cpu()),
                "cached_affected_token_fraction": float((affected * response_mask).sum().item() / response_mask.sum().clamp_min(1.0).item()),
            }
        )
    return diagnostics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    print("Cached PPO rollouts:", len(rows))
    print("Required epsilon values:", cfg["clip_values"])
    print("Cache keys:", sorted(rows[0].keys()))
    diagnostics = cached_clipping_diagnostics(rows, cfg)
    save_json(f"{cfg['results_dir']}/clipping_cached_diagnostics.json", {"cached_rollouts": len(rows), "epsilons": diagnostics})
    for item in diagnostics:
        print(item)

    fork_results = []
    for eps in cfg["clip_values"]:
        run_name = f"clip_{float(eps):.2f}".replace(".", "_")
        output = f"outputs/task2_ppo/{run_name}"
        run_ppo(
            args.config,
            output=output,
            updates=int(cfg["fork_updates"]),
            clip_epsilon=float(eps),
            kl_beta=float(cfg["kl_beta"]),
            run_name=run_name,
        )
        eval_result = evaluate_policy(args.config, output, run_name)
        fork_results.append(eval_result)
    save_json(f"{cfg['results_dir']}/clipping_fork_summary.json", {"cached": diagnostics, "forks": fork_results})


if __name__ == "__main__":
    main()
