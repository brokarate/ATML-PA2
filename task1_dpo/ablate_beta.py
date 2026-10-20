from __future__ import annotations

import argparse
from common.data import load_yaml
from task1_dpo.evaluate import evaluate_adapter
from task1_dpo.train import run_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Required beta values:", cfg["betas"])
    print("Short-run examples per condition:", cfg["short_ablation_examples"])
    for beta in cfg["betas"]:
        run_name = f"beta_{float(beta):.2f}".replace(".", "_")
        output = f"outputs/task1_dpo/{run_name}"
        print(f"\n=== Training {run_name} ===")
        run_training(
            args.config,
            run_name,
            dataset_path=cfg["paths"]["dpo_standard_train"],
            output_path=output,
            beta=float(beta),
            max_examples=int(cfg["short_ablation_examples"]),
        )
        print(f"\n=== Evaluating {run_name} ===")
        evaluate_adapter(args.config, output, run_name)


if __name__ == "__main__":
    main()
