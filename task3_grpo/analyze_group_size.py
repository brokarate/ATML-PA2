from __future__ import annotations

import argparse
from collections import defaultdict
import numpy as np

from common.data import load_yaml, read_jsonl


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int):
    """Return K-sized groups while keeping total cached completions fixed.

    Students should decide and document exactly how prompts/completions are partitioned for the
    requested equal-generation comparison.
    """
    groups = []
    for prompt_id, completions in by_prompt.items():
        usable = completions[:8]
        for start in range(0, len(usable), k):
            chunk = usable[start : start + k]
            if len(chunk) == k:
                groups.append((prompt_id, chunk))
    return groups


def _group_stats(groups):
    rewards = []
    stds = []
    adv_values = []
    informative = 0
    for _, group in groups:
        vals = np.asarray([float(row["reward"]) for row in group], dtype=float)
        rewards.append(float(vals.mean()))
        std = float(vals.std())
        stds.append(std)
        informative += int(std > 1e-6)
        adv = (vals - vals.mean()) / (std + 1e-6)
        adv_values.extend(float(x) for x in adv)
    return {
        "groups": len(groups),
        "informative_group_rate": float(informative / max(len(groups), 1)),
        "mean_group_reward_std": float(np.mean(stds)) if stds else 0.0,
        "relative_signal_variance": float(np.var(adv_values)) if adv_values else 0.0,
        "mean_group_reward": float(np.mean(rewards)) if rewards else 0.0,
    }


def _difficulty_bins(by_prompt):
    means = {
        prompt_id: float(np.mean([float(row["reward"]) for row in group[:8]]))
        for prompt_id, group in by_prompt.items()
    }
    median = float(np.median(list(means.values())))
    return {
        "low_reward": {pid for pid, value in means.items() if value <= median},
        "high_reward": {pid for pid, value in means.items() if value > median},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    print("Cached prompts:", len(by_prompt))
    print("Group sizes to analyze:", cfg["group_sizes"])
    first = next(iter(by_prompt.values()))
    print("Cache row keys:", sorted(first[0].keys()))
    bins = _difficulty_bins(by_prompt)
    summary = {
        "difficulty_binning": "low_reward <= median prompt mean reward; high_reward > median",
        "group_sizes": {},
    }
    for k in cfg["group_sizes"]:
        groups = regroup_equal_generation_budget(by_prompt, int(k))
        item = _group_stats(groups)
        item["difficulty_bins"] = {}
        for name, prompt_ids in bins.items():
            item["difficulty_bins"][name] = _group_stats([(pid, group) for pid, group in groups if pid in prompt_ids])
        summary["group_sizes"][str(k)] = item
        print("K=", k, item)

    from common.logging_utils import save_json

    save_json(f"{cfg['results_dir']}/group_size_summary.json", summary)


if __name__ == "__main__":
    main()
