from __future__ import annotations

import argparse
from common.data import load_yaml
from common.logging_utils import save_json
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import evaluate_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("KL beta conditions:", cfg["kl_values"])
    print("Fork update budget:", cfg["fork_updates"])
    results = []
    for beta in cfg["kl_values"]:
        run_name = f"kl_{float(beta):.2f}".replace(".", "_")
        output = f"outputs/task2_ppo/{run_name}"
        train_summary = run_ppo(
            args.config,
            output=output,
            updates=int(cfg["fork_updates"]),
            clip_epsilon=float(cfg["clip_epsilon"]),
            kl_beta=float(beta),
            run_name=run_name,
        )
        eval_summary = evaluate_policy(args.config, output, run_name)
        results.append({"train": train_summary, "eval": eval_summary})
    save_json(f"{cfg['results_dir']}/kl_ablation_summary.json", {"kl_values": cfg["kl_values"], "runs": results})


if __name__ == "__main__":
    main()
