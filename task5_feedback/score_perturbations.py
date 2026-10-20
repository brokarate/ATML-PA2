from __future__ import annotations

import argparse
from collections import defaultdict

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}

COMPARISONS = {
    "reasoning_sensitivity": ("clean_correct", "corrupt_reasoning_correct_final"),
    "outcome_sensitivity": ("clean_correct", "good_reasoning_wrong_final"),
    "filler_susceptibility": ("clean_correct", "persuasive_filler_correct"),
    "distractor_robustness": ("clean_correct", "gold_distractor_wrong_final"),
    "final_vs_reasoning": ("corrupt_reasoning_correct_final", "good_reasoning_wrong_final"),
}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def _outdir(cfg):
    return repo_path(cfg["results_dir"]) / "task5_feedback"


def _preference_for_variant(pref: str, left: str, right: str):
    return {"A": left, "B": right, "TIE": "tie"}[pref]


def _score_variants(groups):
    rows = []
    for problem_id, variants in sorted(groups.items()):
        for variant_type, row in sorted(variants.items()):
            actual = exact_reward(row["response"], str(row["gold_final"]))
            rows.append(
                {
                    "problem_id": problem_id,
                    "variant_type": variant_type,
                    "expected_exact_reward": float(row["expected_exact_reward"]),
                    "exact_reward": actual,
                    "matches_expected": actual == float(row["expected_exact_reward"]),
                    "reasoning_quality": row.get("reasoning_quality"),
                    "style": row.get("style"),
                }
            )
    return rows


def _compare_variants(cfg, groups):
    outdir = _outdir(cfg)
    detail_path = outdir / "controlled_pairwise.jsonl"
    existing = read_jsonl(detail_path) if detail_path.exists() else []
    done = {(row["problem_id"], row["comparison"]) for row in existing}
    judge = None
    new_rows = []
    todo = [
        (problem_id, comparison_name, left, right)
        for problem_id, variants in sorted(groups.items())
        for comparison_name, (left, right) in COMPARISONS.items()
        if (problem_id, comparison_name) not in done
    ]
    if todo:
        judge = PairwiseAIJudge(cfg, outdir / "pairwise_judge_cache.json")

    for idx, (problem_id, comparison_name, left, right) in enumerate(todo, start=1):
        variants = groups[problem_id]
        left_row = variants[left]
        right_row = variants[right]
        pref = judge.compare(left_row["question"], left_row["response"], right_row["response"])
        new_rows.append(
            {
                "problem_id": problem_id,
                "comparison": comparison_name,
                "left_variant": left,
                "right_variant": right,
                "judge_preference": pref,
                "preferred_variant": _preference_for_variant(pref, left, right),
                "left_exact_reward": exact_reward(left_row["response"], str(left_row["gold_final"])),
                "right_exact_reward": exact_reward(right_row["response"], str(right_row["gold_final"])),
                "left_expected_exact_reward": float(left_row["expected_exact_reward"]),
                "right_expected_exact_reward": float(right_row["expected_exact_reward"]),
            }
        )
        if idx % 20 == 0:
            write_jsonl(detail_path, existing + new_rows)
            print(f"controlled pairwise: {idx}/{len(todo)}")

    all_rows = existing + new_rows
    write_jsonl(detail_path, all_rows)
    return all_rows


def _summarize_variant_scores(rows):
    total = len(rows)
    by_variant = {}
    for variant in sorted({row["variant_type"] for row in rows}):
        sub = [row for row in rows if row["variant_type"] == variant]
        by_variant[variant] = {
            "examples": len(sub),
            "exact_reward_rate": sum(row["exact_reward"] for row in sub) / len(sub),
            "matches_expected_rate": sum(row["matches_expected"] for row in sub) / len(sub),
        }
    return {
        "examples": total,
        "overall_matches_expected_rate": sum(row["matches_expected"] for row in rows) / total if total else 0.0,
        "by_variant": by_variant,
    }


def _summarize_pairwise(rows):
    out = {}
    for comparison in sorted({row["comparison"] for row in rows}):
        sub = [row for row in rows if row["comparison"] == comparison]
        n = len(sub)
        prefs = defaultdict(int)
        exact_agrees = 0
        for row in sub:
            prefs[row["preferred_variant"]] += 1
            if row["left_exact_reward"] > row["right_exact_reward"]:
                exact_winner = row["left_variant"]
            elif row["right_exact_reward"] > row["left_exact_reward"]:
                exact_winner = row["right_variant"]
            else:
                exact_winner = "tie"
            exact_agrees += row["preferred_variant"] == exact_winner
        out[comparison] = {
            "comparisons": n,
            "preferred_variant_counts": dict(prefs),
            "judge_exact_agreement_rate": exact_agrees / n if n else 0.0,
            "ai_better_rate": sum(r["preferred_variant"] == r["left_variant"] for r in sub) / n,
            "ai_tie_rate": sum(r["preferred_variant"] == "tie" for r in sub) / n,
            "ai_wrong_rate": sum(r["preferred_variant"] == r["right_variant"] for r in sub) / n,
            "exact_better_rate": sum(r["left_exact_reward"] > r["right_exact_reward"] for r in sub) / n,
            "exact_tie_rate": sum(r["left_exact_reward"] == r["right_exact_reward"] for r in sub) / n,
            "exact_wrong_rate": sum(r["left_exact_reward"] < r["right_exact_reward"] for r in sub) / n,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    outdir = _outdir(cfg)
    outdir.mkdir(parents=True, exist_ok=True)
    variant_rows = _score_variants(groups)
    pairwise_rows = _compare_variants(cfg, groups)
    write_jsonl(outdir / "controlled_exact_scores.jsonl", variant_rows)
    summary = {
        "diagnostic_problems": len(groups),
        "variants_per_problem": sorted(EXPECTED_VARIANTS),
        "exact_verifier": _summarize_variant_scores(variant_rows),
        "pairwise_ai_judge": _summarize_pairwise(pairwise_rows),
        "exact_scores_file": str(outdir / "controlled_exact_scores.jsonl"),
        "pairwise_file": str(outdir / "controlled_pairwise.jsonl"),
    }
    pairs = summary["pairwise_ai_judge"]
    summary["strict_sensitivity"] = [
        {"feedback": feedback,
         "S_reason": pairs["reasoning_sensitivity"][prefix + "_better_rate"],
         "S_outcome": pairs["outcome_sensitivity"][prefix + "_better_rate"]}
        for feedback, prefix in [("exact_verifier", "exact"), ("ai_judge", "ai")]
    ]
    save_json(outdir / "controlled_diagnostics_summary.json", summary)
    print("Wrote:", outdir / "controlled_exact_scores.jsonl")
    print("Wrote:", outdir / "controlled_pairwise.jsonl")
    print("Wrote:", outdir / "controlled_diagnostics_summary.json")


if __name__ == "__main__":
    main()
