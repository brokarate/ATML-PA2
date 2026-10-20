from __future__ import annotations

import ast
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pa2")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "report" / "figures"


POLICY_ORDER = ["sft", "dpo", "ppo", "grpo"]
POLICY_COLORS = {
    "sft": "#4C78A8",
    "dpo": "#F58518",
    "ppo": "#54A24B",
    "grpo": "#B279A2",
    "rlaif": "#E45756",
    "rlvr": "#72B7B2",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def savefig(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_table(df: pd.DataFrame, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    df.to_csv(FIGURES / f"{name}.csv", index=False)


def pct_axis(ax: plt.Axes) -> None:
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])


def style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def plot_task1() -> None:
    runs = ["standard", "beta_0_03", "beta_0_10", "beta_0_30"]
    rows = []
    for run in runs:
        summary = read_json(RESULTS / "task1_dpo" / run / "summary.json")
        eval_summary = read_json(RESULTS / "task1_dpo" / run / "eval_summary.json")
        rows.append(
            {
                "run": run,
                "beta": summary["beta"],
                "budget_type": "full epoch" if run == "standard" else "short fork",
                "train_examples": summary["train_examples"],
                "optimizer_steps": summary["optimizer_steps"],
                "skipped_train_prompts": summary["skipped_overlong_prompts"],
                "eval_examples": eval_summary["preference"]["examples"],
                "skipped_eval_prompts": eval_summary["skipped_overlong_prompts"],
                "heldout_dpo_loss": eval_summary["preference"]["loss"],
                "preference_accuracy": eval_summary["preference"]["preference_accuracy"],
                "sampled_kl": eval_summary["generation"]["sampled_kl"],
                "reward_model_score": eval_summary["generation"]["reward_model_score"],
                "mean_response_length": eval_summary["generation"]["mean_response_length"],
                "wall_time_min": summary["wall_time_seconds"] / 60,
            }
        )
    dpo = pd.DataFrame(rows)
    save_table(dpo, "task1_dpo_summary_table")

    beta = dpo[dpo["run"].str.startswith("beta")].sort_values("beta")
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.0))
    metrics = [
        ("preference_accuracy", "Preference accuracy", True),
        ("reward_model_score", "Reward model score", False),
        ("mean_response_length", "Mean response length", False),
    ]
    for ax, (metric, title, is_pct) in zip(axes, metrics):
        ax.plot(beta["beta"], beta[metric], marker="o", color="#F58518", label="short beta forks")
        ax.axhline(float(dpo.loc[dpo["run"] == "standard", metric].iloc[0]), color="#333333", linestyle="--", linewidth=1.4, label="standard full epoch")
        ax.set_xlabel("DPO beta")
        ax.set_title(title)
        if is_pct:
            pct_axis(ax)
        style_axes(ax)
    axes[0].legend(frameon=False, fontsize=8)
    savefig(fig, "task1_dpo_beta_ablation")

    len_rows = []
    for run, label in [
        ("standard_length_stratified", "standard DPO"),
        ("length_balanced_stratified", "length-balanced DPO"),
    ]:
        eval_summary = read_json(RESULTS / "task1_dpo" / run / "eval_summary.json")
        len_rows.append(
            {
                "run": label,
                "eval_examples": eval_summary["preference"]["examples"],
                "skipped_eval_prompts": eval_summary["skipped_overlong_prompts"],
                "heldout_dpo_loss": eval_summary["preference"]["loss"],
                "preference_accuracy": eval_summary["preference"]["preference_accuracy"],
                "sampled_kl": eval_summary["generation"]["sampled_kl"],
                "reward_model_score": eval_summary["generation"]["reward_model_score"],
                "mean_response_length": eval_summary["generation"]["mean_response_length"],
            }
        )
    length_df = pd.DataFrame(len_rows)
    save_table(length_df, "task1_length_balanced_eval_table")

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.0))
    x = np.arange(len(length_df))
    labels = length_df["run"].tolist()
    for ax, metric, title, is_pct in [
        (axes[0], "preference_accuracy", "Length-stratified pref. accuracy", True),
        (axes[1], "reward_model_score", "Reward model score", False),
        (axes[2], "mean_response_length", "Mean response length", False),
    ]:
        ax.bar(x, length_df[metric], color=["#F58518", "#72B7B2"], width=0.62)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_title(title)
        if is_pct:
            pct_axis(ax)
        style_axes(ax)
    savefig(fig, "task1_length_bias_summary")


def plot_task2() -> None:
    train = read_jsonl(RESULTS / "task2_ppo" / "standard" / "train_log.jsonl")
    save_table(train, "task2_ppo_standard_train_log")

    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.0), sharex=True)
    panels = [
        ("learned_reward", "Reward"),
        ("kl", "KL"),
        ("policy_loss", "Policy loss"),
        ("value_loss", "Value loss"),
        ("entropy", "Entropy"),
        ("clip_fraction", "Clip fraction"),
        ("policy_grad_norm", "Policy grad norm"),
        ("response_length", "Response length"),
    ]
    for ax, (metric, title) in zip(axes.ravel(), panels):
        ax.plot(train["update"], train[metric], marker="o", linewidth=1.5, markersize=3, color="#54A24B")
        ax.set_title(title)
        ax.set_xlabel("PPO update")
        style_axes(ax)
        if metric == "clip_fraction":
            pct_axis(ax)
    savefig(fig, "task2_ppo_standard_trajectory")

    clipping = read_json(RESULTS / "task2_ppo" / "clipping_fork_summary.json")
    cached = pd.DataFrame(clipping["cached"])
    forks = pd.DataFrame(clipping["forks"])
    forks["clip_epsilon"] = forks["name"].str.extract(r"clip_0_(\d+)").astype(float) / 100
    clip_df = cached.merge(forks, on="clip_epsilon", how="left")
    save_table(clip_df, "task2_clipping_ablation_table")

    kl = read_json(RESULTS / "task2_ppo" / "kl_ablation_summary.json")
    kl_rows = []
    for item in kl["runs"]:
        row = {"kl_beta": item["train"]["kl_beta"], "wall_time_seconds": item["train"]["wall_time_seconds"]}
        row.update(item["eval"])
        kl_rows.append(row)
    kl_df = pd.DataFrame(kl_rows)
    save_table(kl_df, "task2_kl_ablation_table")

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.0))
    axes[0, 0].plot(clip_df["clip_epsilon"], clip_df["cached_clip_fraction"], marker="o", label="clip fraction", color="#E45756")
    axes[0, 0].plot(clip_df["clip_epsilon"], clip_df["cached_affected_token_fraction"], marker="s", label="affected token fraction", color="#4C78A8")
    axes[0, 0].set_title("Cached clipping diagnostics")
    axes[0, 0].set_xlabel("clip epsilon")
    pct_axis(axes[0, 0])
    axes[0, 0].legend(frameon=False, fontsize=8)
    for ax, df, xcol, metric, title, color in [
        (axes[0, 1], clip_df, "clip_epsilon", "reward_model_score", "Clipping fork reward", "#54A24B"),
        (axes[0, 2], clip_df, "clip_epsilon", "mean_response_length", "Clipping fork length", "#B279A2"),
        (axes[1, 0], kl_df, "kl_beta", "reward_model_score", "KL beta reward", "#54A24B"),
        (axes[1, 1], kl_df, "kl_beta", "sampled_kl", "KL beta sampled KL", "#F58518"),
        (axes[1, 2], kl_df, "kl_beta", "mean_response_length", "KL beta length", "#B279A2"),
    ]:
        ax.plot(df[xcol], df[metric], marker="o", color=color)
        ax.set_title(title)
        ax.set_xlabel(xcol.replace("_", " "))
        style_axes(ax)
    for ax in axes.ravel():
        style_axes(ax)
    savefig(fig, "task2_ppo_ablations")


def plot_task3() -> None:
    train = read_jsonl(RESULTS / "task3_grpo" / "standard" / "train_log.jsonl")
    save_table(train, "task3_grpo_standard_train_log")

    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.0), sharex=True)
    panels = [
        ("reward", "Reward"),
        ("kl", "KL"),
        ("group_reward_std", "Group reward std"),
        ("uninformative_group_fraction", "Uninformative groups"),
        ("loss", "Policy loss"),
        ("grad_norm", "Grad norm"),
        ("entropy", "Entropy"),
        ("response_length", "Response length"),
    ]
    for ax, (metric, title) in zip(axes.ravel(), panels):
        ax.plot(train["update"], train[metric], marker="o", linewidth=1.5, markersize=3, color="#B279A2")
        ax.set_title(title)
        ax.set_xlabel("GRPO update")
        style_axes(ax)
        if metric == "uninformative_group_fraction":
            pct_axis(ax)
    savefig(fig, "task3_grpo_standard_trajectory")

    group = read_json(RESULTS / "task3_grpo" / "group_size_summary.json")
    group_rows = []
    for k, payload in group["group_sizes"].items():
        group_rows.append({"K": int(k), "bin": "overall", **{m: payload[m] for m in ["groups", "informative_group_rate", "mean_group_reward_std", "relative_signal_variance", "mean_group_reward"]}})
        for bin_name, bin_payload in payload["difficulty_bins"].items():
            group_rows.append({"K": int(k), "bin": bin_name, **bin_payload})
    group_df = pd.DataFrame(group_rows).sort_values(["bin", "K"])
    save_table(group_df, "task3_group_size_table")

    norm = read_json(RESULTS / "task3_grpo" / "normalization_comparison_summary.json")
    norm_rows = []
    for item in norm["runs"]:
        row = {"run": item["train"]["run"], "loss_type": item["train"]["loss_type"], "wall_time_seconds": item["train"]["wall_time_seconds"]}
        row.update(item["eval"])
        norm_rows.append(row)
    standard_eval = read_json(RESULTS / "task3_grpo" / "standard" / "eval_summary.json")
    norm_rows.insert(
        0,
        {
            "run": "standard",
            "loss_type": "grpo",
            "wall_time_seconds": float(train["wall_time_seconds"].max()),
            **standard_eval,
        },
    )
    norm_df = pd.DataFrame(norm_rows)
    save_table(norm_df, "task3_normalization_eval_table")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    for bin_name, style, color in [
        ("overall", "o-", "#333333"),
        ("low_reward", "s--", "#E45756"),
        ("high_reward", "^--", "#4C78A8"),
    ]:
        sub = group_df[group_df["bin"] == bin_name].sort_values("K")
        axes[0].plot(sub["K"], sub["informative_group_rate"], style, color=color, label=bin_name)
        axes[1].plot(sub["K"], sub["mean_group_reward_std"], style, color=color, label=bin_name)
    axes[0].set_title("Informative-group rate")
    axes[1].set_title("Within-group reward std")
    pct_axis(axes[0])
    axes[0].legend(frameon=False, fontsize=8)
    x = np.arange(len(norm_df))
    axes[2].bar(x, norm_df["reward_model_score"], color=["#B279A2", "#54A24B", "#F58518"])
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(norm_df["run"], rotation=18, ha="right")
    axes[2].set_title("Normalization held-out reward")
    for ax in axes:
        ax.set_xlabel("Group size K" if ax is not axes[2] else "")
        style_axes(ax)
    savefig(fig, "task3_group_size_and_normalization")


def plot_task4() -> None:
    summary = read_json(RESULTS / "task4_safety" / "summary.json")
    rows = []
    for policy, payload in summary["policies"].items():
        rows.append({"policy": policy, **payload})
    df = pd.DataFrame(rows)
    df["policy"] = pd.Categorical(df["policy"], categories=POLICY_ORDER, ordered=True)
    df = df.sort_values("policy")
    save_table(df.drop(columns=["label_distribution"]), "task4_safety_summary_table")

    metrics = [
        ("safe_answer_rate", "Safe answer"),
        ("over_refusal_rate", "Over-refusal"),
        ("justified_refusal_rate", "Justified refusal"),
        ("unsafe_compliance_rate", "Unsafe compliance"),
        ("ambiguous_rate", "Ambiguous"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), gridspec_kw={"width_ratios": [3.2, 1]})
    x = np.arange(len(df))
    width = 0.15
    for i, (metric, label) in enumerate(metrics):
        axes[0].bar(x + (i - 2) * width, df[metric], width=width, label=label)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["policy"])
    axes[0].set_title("Safety behavior by policy")
    pct_axis(axes[0])
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[1].bar(df["policy"], df["mean_response_tokens"], color=[POLICY_COLORS.get(p, "#777777") for p in df["policy"].astype(str)])
    axes[1].set_title("Mean response tokens")
    for ax in axes:
        style_axes(ax)
    savefig(fig, "task4_safety_policy_summary")

    cat = pd.read_csv(RESULTS / "task4_safety" / "category_metrics.csv")
    cat["policy"] = pd.Categorical(cat["policy"], categories=POLICY_ORDER, ordered=True)
    cat = cat.sort_values(["type", "policy"])
    save_table(cat, "task4_category_metrics_table")

    safe = cat[cat["safe_examples"] > 0].pivot(index="type", columns="policy", values="over_refusal_rate")
    unsafe = cat[cat["unsafe_examples"] > 0].pivot(index="type", columns="policy", values="unsafe_compliance_rate")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 7.2), gridspec_kw={"width_ratios": [1, 1]})
    for ax, data, title in [
        (axes[0], safe, "Safe prompts: over-refusal rate"),
        (axes[1], unsafe, "Unsafe prompts: unsafe-compliance rate"),
    ]:
        im = ax.imshow(data.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(0.5, float(np.nanmax(data.values))))
        ax.set_xticks(np.arange(len(data.columns)))
        ax.set_xticklabels(data.columns)
        ax.set_yticks(np.arange(len(data.index)))
        ax.set_yticklabels(data.index, fontsize=7)
        ax.set_title(title)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data.values[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6, color="#222222")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig(fig, "task4_category_heatmaps")


def plot_task5() -> None:
    gsm = read_json(RESULTS / "task5_feedback" / "gsm_summary.json")
    transfer = read_json(RESULTS / "task5_feedback" / "transfer_summary.json")
    rows = []
    for dataset, payload in [("GSM8K", gsm), ("SVAMP", transfer)]:
        for policy, metrics in payload["policies"].items():
            rows.append({"dataset": dataset, "policy": policy, **metrics})
    metrics_df = pd.DataFrame(rows)
    save_table(metrics_df, "task5_policy_metrics_table")

    pivot_acc = metrics_df.pivot(index="policy", columns="dataset", values="exact_accuracy").reset_index()
    pivot_acc["accuracy_drop_gsm_to_svamp"] = pivot_acc["GSM8K"] - pivot_acc["SVAMP"]
    save_table(pivot_acc, "task5_accuracy_drop_table")

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))
    policies = ["sft", "rlvr", "rlaif"]
    x = np.arange(len(policies))
    width = 0.35
    for offset, dataset in [(-width / 2, "GSM8K"), (width / 2, "SVAMP")]:
        sub = metrics_df[metrics_df["dataset"] == dataset].set_index("policy").loc[policies]
        axes[0].bar(x + offset, sub["exact_accuracy"], width=width, label=dataset)
        axes[1].bar(x + offset, sub["designated_final_rate"], width=width, label=dataset)
        axes[2].bar(x + offset, sub["mean_response_tokens"], width=width, label=dataset)
    for ax, title, pct in [
        (axes[0], "Exact accuracy", True),
        (axes[1], "Format compliance", True),
        (axes[2], "Mean response tokens", False),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(policies)
        ax.set_title(title)
        if pct:
            pct_axis(ax)
        ax.legend(frameon=False, fontsize=8)
        style_axes(ax)
    savefig(fig, "task5_gsm_svamp_policy_comparison")

    pairwise_rows = []
    for dataset, payload in [("GSM8K", gsm), ("SVAMP", transfer)]:
        for comparison, comp in payload["pairwise"].items():
            row = {"dataset": dataset, "comparison": comparison, "comparisons": comp["comparisons"], "judge_exact_agreement_rate": comp["judge_exact_agreement_rate"]}
            for key, value in comp["winner_counts"].items():
                row[f"wins_{key}"] = value
            pairwise_rows.append(row)
    save_table(pd.DataFrame(pairwise_rows), "task5_pairwise_judge_table")

    controlled = read_jsonl(RESULTS / "task5_feedback" / "controlled_pairwise.jsonl")
    intended_better = {
        "distractor_robustness": "clean_correct",
        "filler_susceptibility": "clean_correct",
        "final_vs_reasoning": "corrupt_reasoning_correct_final",
        "outcome_sensitivity": "clean_correct",
        "reasoning_sensitivity": "clean_correct",
    }
    diag_rows = []
    for comparison, sub in controlled.groupby("comparison"):
        total = len(sub)
        target = intended_better[comparison]
        better = int((sub["preferred_variant"] == target).sum())
        ties = int((sub["preferred_variant"] == "tie").sum())
        wrong = total - better - ties

        exact_better = 0
        exact_tie = 0
        exact_wrong = 0
        for _, row in sub.iterrows():
            target_reward = row["left_exact_reward"] if row["left_variant"] == target else row["right_exact_reward"]
            other_reward = row["right_exact_reward"] if row["left_variant"] == target else row["left_exact_reward"]
            if target_reward > other_reward:
                exact_better += 1
            elif target_reward == other_reward:
                exact_tie += 1
            else:
                exact_wrong += 1

        diag_rows.append(
            {
                "comparison": comparison,
                "intended_better_variant": target,
                "ai_better_rate": better / total,
                "ai_tie_rate": ties / total,
                "ai_wrong_rate": wrong / total,
                "exact_better_rate": exact_better / total,
                "exact_tie_rate": exact_tie / total,
                "exact_wrong_rate": exact_wrong / total,
                "comparisons": total,
            }
        )
    diag = pd.DataFrame(diag_rows).sort_values("comparison")
    save_table(diag, "task5_controlled_pairwise_rates")

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    x = np.arange(len(diag))
    bottom = np.zeros(len(diag))
    stacks = [
        ("ai_better_rate", "AI chose intended-better", "#54A24B"),
        ("ai_tie_rate", "AI tie", "#B9B9B9"),
        ("ai_wrong_rate", "AI chose intended-worse", "#E45756"),
    ]
    for col, label, color in stacks:
        ax.bar(x, diag[col], bottom=bottom, label=label, color=color)
        bottom += diag[col].to_numpy()
    ax.set_xticks(x)
    ax.set_xticklabels(diag["comparison"], rotation=22, ha="right")
    ax.set_title("Controlled feedback diagnostics")
    pct_axis(ax)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    style_axes(ax)
    savefig(fig, "task5_controlled_feedback_diagnostics")


def write_evidence_status() -> None:
    rows = [
        {"task": "1", "requirement": "DPO summary: beta/loss/preference accuracy/KL/reward/length", "artifact": "task1_dpo_summary_table.csv; task1_dpo_beta_ablation.png", "status": "covered from saved results"},
        {"task": "1", "requirement": "Length-confounding diagnostics", "artifact": "task1_length_balanced_eval_table.csv; task1_length_bias_summary.png", "status": "aggregate covered; per-stratum saved metrics and word-limit compliance are not saved"},
        {"task": "1", "requirement": "Qualitative examples", "artifact": "not generated as prose", "status": "requires manual selection/interpretation from saved generations"},
        {"task": "2", "requirement": "PPO continuation trajectories", "artifact": "task2_ppo_standard_trajectory.png", "status": "covered from train_log"},
        {"task": "2", "requirement": "Clipping and KL ablations", "artifact": "task2_ppo_ablations.png; task2_clipping_ablation_table.csv; task2_kl_ablation_table.csv", "status": "covered from saved summaries"},
        {"task": "3", "requirement": "GRPO continuation trajectories", "artifact": "task3_grpo_standard_trajectory.png", "status": "covered from train_log"},
        {"task": "3", "requirement": "Group size and normalization comparison", "artifact": "task3_group_size_and_normalization.png; task3_group_size_table.csv; task3_normalization_eval_table.csv", "status": "covered; exact length-conditioned gradient statistic not separately saved"},
        {"task": "4", "requirement": "Safety calibration by policy", "artifact": "task4_safety_policy_summary.png; task4_safety_summary_table.csv", "status": "covered from judged outputs"},
        {"task": "4", "requirement": "Category-level XSTest behavior", "artifact": "task4_category_heatmaps.png; task4_category_metrics_table.csv", "status": "covered from category metrics"},
        {"task": "4", "requirement": "Manual audit agreement", "artifact": "manual_audit_sheet.csv in results/task4_safety", "status": "not complete until manual_label values are filled"},
        {"task": "5", "requirement": "GSM/SVAMP policy comparison and pairwise judge", "artifact": "task5_gsm_svamp_policy_comparison.png; task5_policy_metrics_table.csv; task5_pairwise_judge_table.csv", "status": "covered from saved results"},
        {"task": "5", "requirement": "Controlled perturbation diagnostics", "artifact": "task5_controlled_feedback_diagnostics.png; task5_controlled_pairwise_rates.csv", "status": "covered from saved diagnostics"},
    ]
    extra = FIGURES / "task1_strata_and_word_limits_verified.csv"
    rows[1].update({"artifact": "task1_strata_and_word_limits_verified.csv; task1_length_balanced_eval_table.csv",
                    "status": "covered by separate frozen-checkpoint evaluation" if extra.exists() else "frozen-checkpoint evaluation pending"})
    rows[4]["artifact"] += "; task2_stability_statistics.csv"
    rows[6].update({"artifact": rows[6]["artifact"] + "; task3_length_conditioned_normalization.csv",
                    "status": "held-out fork metrics plus post-hoc K=4 regrouped-cache log-probability-gradient diagnostic; not historical parameter gradients"})
    rows[8]["artifact"] += "; task4_category_label_distribution.png"
    rows[9].update({"artifact":"task4_manual_audit_confusion.png; results/task4_safety/verified_audit_analysis.json",
                    "status":"240 human labels complete; exact raw-label agreement 0.633333"})
    rows[11]["artifact"] += "; task5_sensitivity_verified.csv"
    rows.append({"task":"2/3", "requirement":"Peak memory", "artifact":"standard/summary.json",
                 "status":"MPS memory was not recorded; zero is a CUDA-only placeholder, not a measurement"})
    df = pd.DataFrame(rows)
    save_table(df, "evidence_status")
    lines = [
        "# PA2 Evidence Artifacts",
        "",
        "Generated from saved experiment outputs only. No model training, model loading, or new evaluation is performed by this plotting script.",
        "",
    ]
    for row in rows:
        lines.append(f"- Task {row['task']}: {row['requirement']} -> {row['artifact']} ({row['status']})")
    (FIGURES / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 130,
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_task1()
    plot_task2()
    plot_task3()
    plot_task4()
    plot_task5()
    write_evidence_status()
    # Rebuild post-hoc diagnostics and the additional Task 4 figures from saved data.
    from scripts.complete_saved_evidence import main as complete_saved_evidence
    complete_saved_evidence()
    print(f"Wrote report figures and tables to {FIGURES}")


if __name__ == "__main__":
    main()
