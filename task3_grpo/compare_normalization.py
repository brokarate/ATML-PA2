from __future__ import annotations

import argparse
from common.data import load_yaml
from common.logging_utils import save_json
from task3_grpo.continue_train import run_grpo
from task3_grpo.evaluate import evaluate_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Fork updates:", cfg["fork_updates"])
    print("Compare loss_type='grpo' vs loss_type='dr_grpo' from the identical supplied midpoint.")
    results = []
    for loss_type in ("grpo", "dr_grpo"):
        run_name = f"{loss_type}_norm"
        output = f"outputs/task3_grpo/{run_name}"
        train_summary = run_grpo(
            args.config,
            output=output,
            updates=int(cfg["fork_updates"]),
            loss_type=loss_type,
            run_name=run_name,
        )
        eval_summary = evaluate_policy(args.config, output, run_name)
        results.append({"train": train_summary, "eval": eval_summary})
    save_json(f"{cfg['results_dir']}/normalization_comparison_summary.json", {"runs": results})


if __name__ == "__main__":
    main()
