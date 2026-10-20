from __future__ import annotations

import argparse

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json


def _outdir(cfg):
    return repo_path(cfg["results_dir"]) / "task5_feedback"


def _maybe_load(path):
    path = repo_path(path)
    return load_json(path) if path.exists() else None


def _policy_table(summary):
    if not summary:
        return {}
    return {
        policy: {
            "exact_accuracy": metrics["exact_accuracy"],
            "designated_final_rate": metrics["designated_final_rate"],
            "mean_response_tokens": metrics["mean_response_tokens"],
            "truncated_rate": metrics["truncated_rate"],
        }
        for policy, metrics in summary["policies"].items()
    }


def _best_policy(policy_table):
    if not policy_table:
        return None
    return max(policy_table, key=lambda p: policy_table[p]["exact_accuracy"])


def _write_markdown(path, combined):
    lines = [
        "# Task 5 Feedback Comparison",
        "",
        "## In-domain GSM8K",
    ]
    for policy, metrics in combined.get("gsm_policy_metrics", {}).items():
        lines.append(
            f"- {policy}: exact accuracy {metrics['exact_accuracy']:.3f}, "
            f"designated-final rate {metrics['designated_final_rate']:.3f}, "
            f"mean response tokens {metrics['mean_response_tokens']:.1f}"
        )
    lines += ["", "## Transfer Math"]
    for policy, metrics in combined.get("transfer_policy_metrics", {}).items():
        lines.append(
            f"- {policy}: exact accuracy {metrics['exact_accuracy']:.3f}, "
            f"designated-final rate {metrics['designated_final_rate']:.3f}, "
            f"mean response tokens {metrics['mean_response_tokens']:.1f}"
        )
    lines += ["", "## Controlled Diagnostics"]
    controlled = combined.get("controlled_diagnostics") or {}
    for name, metrics in (controlled.get("pairwise_ai_judge") or {}).items():
        lines.append(
            f"- {name}: judge/exact agreement {metrics['judge_exact_agreement_rate']:.3f}; "
            f"preferences {metrics['preferred_variant_counts']}"
        )
    lines += ["", "## Notes", combined.get("notes", "")]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = _outdir(cfg)
    gsm = _maybe_load(outdir / "gsm_summary.json")
    transfer = _maybe_load(outdir / "transfer_summary.json")
    controlled = _maybe_load(outdir / "controlled_diagnostics_summary.json")

    missing = []
    if gsm is None:
        missing.append("gsm_summary.json")
    if transfer is None:
        missing.append("transfer_summary.json")
    if controlled is None:
        missing.append("controlled_diagnostics_summary.json")
    if missing:
        raise FileNotFoundError(
            "Run the earlier Task 5 commands first; missing: " + ", ".join(missing)
        )

    gsm_table = _policy_table(gsm)
    transfer_table = _policy_table(transfer)
    combined = {
        "gsm_policy_metrics": gsm_table,
        "transfer_policy_metrics": transfer_table,
        "gsm_pairwise": gsm.get("pairwise", {}),
        "transfer_pairwise": transfer.get("pairwise", {}),
        "controlled_diagnostics": controlled,
        "best_gsm_policy_by_exact_accuracy": _best_policy(gsm_table),
        "best_transfer_policy_by_exact_accuracy": _best_policy(transfer_table),
        "notes": (
            "RLVR is scored by the exact designated-final verifier. RLAIF preferences are scored "
            "by the fixed pairwise AI judge without access to gold answers. Interpret judge-based "
            "comparisons alongside exact accuracy and the controlled perturbation diagnostics."
        ),
    }
    save_json(outdir / "feedback_comparison_summary.json", combined)
    _write_markdown(outdir / "feedback_comparison_summary.md", combined)
    print("Wrote:", outdir / "feedback_comparison_summary.json")
    print("Wrote:", outdir / "feedback_comparison_summary.md")


if __name__ == "__main__":
    main()
