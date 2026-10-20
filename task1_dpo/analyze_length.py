from __future__ import annotations

import argparse
from collections import defaultdict

from common.data import load_yaml, read_jsonl
from common.logging_utils import save_json
from task1_dpo.evaluate import evaluate_adapter
from task1_dpo.train import run_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    balanced = read_jsonl(cfg["paths"]["dpo_length_train"])
    stratified = read_jsonl(cfg["paths"]["dpo_length_eval"])
    print("Length-balanced train rows:", len(balanced))
    print("Length-stratified eval rows:", len(stratified))
    run_training(
        args.config,
        "length_balanced",
        dataset_path=cfg["paths"]["dpo_length_train"],
        output_path=cfg["length_output"],
        beta=float(cfg["beta"]),
    )
    standard = evaluate_adapter(
        args.config,
        cfg["standard_output"],
        "standard_length_stratified",
        eval_path=cfg["paths"]["dpo_length_eval"],
    )
    length_balanced = evaluate_adapter(
        args.config,
        cfg["length_output"],
        "length_balanced_stratified",
        eval_path=cfg["paths"]["dpo_length_eval"],
    )
    strata = defaultdict(int)
    for row in stratified:
        strata[str(row.get("length_stratum", row.get("stratum", row.get("category", "unknown"))))] += 1
    save_json(
        f"{cfg['results_dir']}/length_confounding_summary.json",
        {
            "strata_counts": dict(strata),
            "standard": standard,
            "length_balanced": length_balanced,
        },
    )


if __name__ == "__main__":
    main()
