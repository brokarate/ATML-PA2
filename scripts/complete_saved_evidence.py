"""Rebuild missing diagnostics and figures from saved data without model loading."""
from __future__ import annotations

import csv
import argparse
import importlib.metadata
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pa2")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common.data import read_jsonl, repo_path
from common.logging_utils import save_json
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss

LABELS = ["SAFE_ANSWER", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "OVER_REFUSAL", "AMBIGUOUS"]
FIG = repo_path("report/figures")
OUT = repo_path("results/completion_audit")


def table(name, rows):
    path = FIG / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def saved_metrics():
    dispersion, stability = [], []
    for task in ("task1_dpo", "task2_ppo", "task3_grpo"):
        for path in sorted(repo_path(f"results/{task}").glob("*/generations.jsonl")):
            rows = read_jsonl(path)
            vals = [r["response_length"] for r in rows]
            dispersion.append({"task": task, "run": path.parent.name, "examples": len(vals),
                "mean_tokens": statistics.mean(vals), "sample_sd_tokens": statistics.stdev(vals),
                "q25_linear": float(np.quantile(vals, .25)), "q75_linear": float(np.quantile(vals, .75)),
                "iqr_linear": float(np.quantile(vals, .75)-np.quantile(vals, .25))})
    safety = read_jsonl("results/task4_safety/generated_all.jsonl")
    for policy in sorted({r["policy"] for r in safety}):
        vals = [r["response_tokens"] for r in safety if r["policy"] == policy]
        dispersion.append({"task":"task4_safety", "run":policy,
            "examples":len(vals),"mean_tokens":statistics.mean(vals),
            "sample_sd_tokens":statistics.stdev(vals),
            "q25_linear":float(np.quantile(vals,.25)),"q75_linear":float(np.quantile(vals,.75)),
            "iqr_linear":float(np.quantile(vals,.75)-np.quantile(vals,.25))})
    for stem in ("gsm", "transfer"):
        rows = read_jsonl(f"results/task5_feedback/{stem}_generations.jsonl")
        for policy in sorted({r["policy"] for r in rows}):
            vals = [r["response_tokens"] for r in rows if r["policy"] == policy]
            dispersion.append({"task":"task5_feedback", "run":f"{stem}/{policy}",
                "examples":len(vals),"mean_tokens":statistics.mean(vals),
                "sample_sd_tokens":statistics.stdev(vals),
                "q25_linear":float(np.quantile(vals,.25)),"q75_linear":float(np.quantile(vals,.75)),
                "iqr_linear":float(np.quantile(vals,.75)-np.quantile(vals,.25))})
    for path in sorted(repo_path("results/task2_ppo").glob("*/train_log.jsonl")):
        rows = read_jsonl(path)
        stability.append({"run": path.parent.name, "updates": len(rows),
            "skipped_updates": sum(r["skipped_update"] for r in rows),
            "max_preclip_policy_grad_norm": max(r["policy_grad_norm"] for r in rows),
            "learned_reward_sample_sd": statistics.stdev(r["learned_reward"] for r in rows)})
    table("response_length_dispersion", dispersion)
    table("task2_stability_statistics", stability)
    return {"response_length_dispersion": dispersion, "ppo_stability": stability}


def task1_tables(export_notes=False):
    path = repo_path("results/task1_dpo/completion_diagnostics/summary.json")
    if not path.exists():
        return {"status":"frozen-checkpoint evaluation not complete"}
    data = json.loads(path.read_text())
    if len(data["runs"]) != 2:
        return {"status":"frozen-checkpoint evaluation in progress"}
    rows = []
    for run, metrics in data["runs"].items():
        r = {"run":run}
        for name in ("preferred_longer","length_matched","rejected_longer"):
            r[name+"_accuracy"] = metrics["strata"][name]["preference_accuracy"]
            r[name+"_examples"] = metrics["strata"][name]["examples"]
        r.update({"word_limit_examples":metrics["word_limit_examples"],
            "word_limit_compliance":metrics["word_limit_compliance"],
            "saved_generation_mean_tokens":metrics["saved_generation_mean_tokens"],
            "saved_generation_sample_sd_tokens":metrics["saved_generation_sample_sd_tokens"]})
        rows.append(r)
    table("task1_strata_and_word_limits_verified", rows)
    tex = [r"\begin{center}", r"\captionof{table}{Frozen-checkpoint post-hoc evaluation: stratum preference accuracy and compliance on the fixed ten word-limit prompts. Original aggregate results above are retained separately.}",
           r"\label{tab:task1-strata}", r"\small", r"\begin{tabular}{lrrrr}", r"\toprule",
           r"Run & Preferred longer & Length matched & Rejected longer & Word-limit compliance \\",r"\midrule"]
    for row in rows:
        vals=[row[s+"_accuracy"] for s in ("preferred_longer","length_matched","rejected_longer")]
        tex.append(row["run"].replace("_",r"\_")+" & "+" & ".join(f"{v:.3f}" for v in vals)+f" & {row['word_limit_compliance']:.3f}"+r" \\")
    tex.extend([r"\bottomrule",r"\end{tabular}",r"\end{center}"])
    if export_notes:
        (FIG/"task1_strata_and_word_limits_verified.tex").write_text("\n".join(tex)+"\n")
    return data


def qualitative_sources(export_notes=False):
    # Exact text and score provenance for independent human interpretation.
    cache = torch.load(repo_path("cached/ppo_rollout.pt"), map_location="cpu",weights_only=False)
    prompts = {r["prompt_id"]:r for r in read_jsonl("data/rl_prompt_pool_train.jsonl")}
    prompts.update({r["prompt_id"]:r for r in read_jsonl("data/rl_prompt_pool_eval.jsonl")})
    scored = []
    for r in cache:
        scored.append({"source":"cached/ppo_rollout.pt (supplied midpoint rollout, not final student policy)",
            "prompt_id":r["prompt_id"], "source_index":r["source_index"],
            "prompt_record":prompts.get(r["prompt_id"]), "response":r["response"],
            "response_tokens":r.get("response_tokens"),"raw_terminal_reward":r["raw_terminal_reward"],
            "effective_terminal_reward":r["effective_terminal_reward"],
            "terminated_with_eos":r.get("terminated_with_eos")})
    save_json(OUT/"exact_scored_ppo_cache_examples.json",scored)
    diag = read_jsonl("data/task5_controlled_reward_diagnostics.jsonl")
    save_json(OUT/"exact_task5_diagnostic_examples.json",diag)
    scored_path = OUT/"scored_final_policy_examples.jsonl"
    if export_notes and scored_path.exists():
        scored = read_jsonl(scored_path)
        lookup = {(r['task'],r['prompt_id'],r['run']):r for r in scored}
        selections = [
            ('Task 1: reward versus one-sentence compliance', 'task1_dpo',
             'e64ed52b2b74c3b8fc2d9e135c7b9c5d7975282d892d5e11ff27ee2bc40cfdbf',
             'standard_length_stratified','length_balanced_stratified',
             'Manual inspection: the higher-scored standard response has two sentences despite the one-sentence instruction; the balanced response has one.'),
            ('Task 1: reward versus character limit', 'task1_dpo',
             'd308da7d7d4348f3021d791f8a1d234ed7c586f74e83b8feebd433073206e494',
             'standard_length_stratified','length_balanced_stratified',
             'Computed compliance: standard has 87 characters; balanced has 159. The prompt caps the tweet at 140; the higher-scored balanced response violates the cap.'),
            ('Task 2: reward and instruction quality agree (matched eight-update forks)', 'task2_ppo',
             '4fa4a88327b6c9c9b36eaeb14ab4c57edb9c414319b4e0eeb237154eccd38177',
             'kl_0_10','kl_0_20',
             'Manual inspection: KL 0.10 supplies the requested five numbered choices; KL 0.20 supplies three rules and asks a question instead.'),
            ('Task 2: reward and instruction quality disagree (matched eight-update forks)', 'task2_ppo',
             '4757a77e3611659864043267aea61c38ebab2f02773047426aecac4307ede8a4',
             'kl_0_10','kl_0_20',
             'Manual inspection: the user says they will provide characters and setting. The higher-scored KL 0.10 response invents these; KL 0.20 asks the user for their information.')]
        lines=['# Individually scored qualitative evidence', '',
            'Exact original response texts with new post-hoc scores from the same frozen reward model. No generation or training. Scores were not present per example in the original evaluation summaries. Selection is illustrative, not a random quality estimate.', '']
        for title, task, pid, left, right, note in selections:
            a,b=lookup[(task,pid,left)],lookup[(task,pid,right)]
            lines += ['## '+title,'','Prompt ID: `'+pid+'`','',
                'Exact prompt:','```text',a['prompt_messages'][-1]['content'],'```','']
            for r in (a,b):
                lines += [f"### {r['run']}",'',f"Source: `{r['source_file']}`",'',
                    f"Reward: {r['reward_model_score']}; saved tokens: {r['response_tokens']}; computed words: {r['word_count']}; computed characters: {r['characters']}",'',
                    '```text',r['response'],'```','']
            lines += [note,'']
        repo_path('report/QUALITATIVE_EVIDENCE_VERIFIED.md').write_text('\n'.join(lines)+'\n')
    return {"ppo_scored_cache_examples":len(cache),"task5_diagnostic_examples":len(diag)}


def grpo_normalization():
    cache = read_jsonl("cached/grpo_k_cache.jsonl")
    groups = defaultdict(list)
    for row in cache:
        groups[row["prompt_id"]].append(row)
    rows, ids = [], []
    for gid, key in enumerate(sorted(groups)):
        group = sorted(groups[key], key=lambda r: r["generation_index"])[:8]
        if len(group) != 8:
            raise ValueError("Expected eight fixed cache completions per prompt")
        rows.extend(group)
        # Partition each K=8 cache group into two K=4 groups, matching training.
        ids.extend([2*gid]*4 + [2*gid+1]*4)
    rewards = torch.tensor([r["reward"] for r in rows], dtype=torch.float64)
    advantages = group_relative_advantages(rewards, torch.tensor(ids))
    cap = 512
    mask = torch.zeros((len(rows), cap), dtype=torch.float64)
    for i, r in enumerate(rows):
        if r["terminated_with_eos"] and r["completion_tokens"] <= cap:
            mask[i, :r["completion_tokens"]] = 1
    lengths = mask.sum(1)
    valid = lengths > 0
    cutoff = float(lengths[valid].median())
    records, details = [], []
    for loss_type in ("grpo", "dr_grpo"):
        new = torch.zeros_like(mask, requires_grad=True)
        zero = torch.zeros_like(mask)
        loss, _ = grpo_policy_loss(new, zero, advantages, mask, zero,
            .2, 0., loss_type=loss_type, max_completion_length=cap)
        grad = torch.autograd.grad(loss, new)[0].abs()
        totals = grad.sum(1)
        means = totals / lengths.clamp_min(1)
        for name, selected in [("short", valid & (lengths <= cutoff)),
                               ("long", valid & (lengths > cutoff))]:
            records.append({"normalization": loss_type, "length_bin": name,
                "examples": int(selected.sum()), "cutoff_tokens": cutoff,
                "mean_tokens": float(lengths[selected].mean()),
                "mean_abs_advantage": float(advantages[selected].abs().mean()),
                "mean_abs_logprob_gradient_per_token": float(means[selected].mean()),
                "mean_sum_abs_logprob_gradient_per_completion": float(totals[selected].mean())})
        for i, row in enumerate(rows):
            details.append({"normalization": loss_type, "prompt_id": row["prompt_id"],
                "generation_index": row["generation_index"], "saved_tokens": row["completion_tokens"],
                "masked": not bool(valid[i]), "advantage": float(advantages[i]),
                "mean_abs_logprob_gradient_per_token": float(means[i]),
                "sum_abs_logprob_gradient": float(totals[i])})
    payload = {"source": "cached/grpo_k_cache.jsonl", "group_size": 4,
        "cache_completions": len(rows), "eligible_completions": int(valid.sum()),
        "masked_completions": int((~valid).sum()), "loss_cap": cap,
        "bin_rule": "median valid completion token length; short <= median, long > median",
        "method": "Autograd derivative of the clipped policy surrogate with respect to each response-token log probability at new=old, beta=0. Same fixed rewards, groups and masks for both normalizations.",
        "scope": "Post-hoc fixed-cache mechanism diagnostic; not parameter-gradient norms measured during training. Each source K=8 cache group is partitioned into two K=4 groups to match the forks. Long/nonterminated responses masked to match the 512-token training loss rule.",
        "rows": records}
    save_json("results/task3_grpo/length_conditioned_normalization.json", payload)
    table("task3_length_conditioned_normalization", records)
    table("task3_length_conditioned_normalization_examples", details)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.0))
    for i, (field, title) in enumerate([
        ("mean_abs_logprob_gradient_per_token", "Mean absolute gradient per token"),
        ("mean_sum_abs_logprob_gradient_per_completion", "Mean absolute gradient sum per completion")]):
        for offset, norm, color in [(-.18, "grpo", "#4C78A8"), (.18, "dr_grpo", "#F58518")]:
            vals = [r[field] for r in records if r["normalization"] == norm]
            axes[i].bar(np.arange(2)+offset, vals, .36, label=norm, color=color)
        axes[i].set_xticks([0,1], ["Short", "Long"])
        axes[i].set_title(title)
        axes[i].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        axes[i].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "task3_length_conditioned_normalization.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return payload


def task5():
    rows = read_jsonl("results/task5_feedback/controlled_pairwise.jsonl")
    rates = []
    for comparison in sorted({r["comparison"] for r in rows}):
        sub = [r for r in rows if r["comparison"] == comparison]
        n = len(sub)
        rates.append({"comparison": comparison, "examples": n,
            "exact_better_rate": sum(r["left_exact_reward"] > r["right_exact_reward"] for r in sub)/n,
            "exact_tie_rate": sum(r["left_exact_reward"] == r["right_exact_reward"] for r in sub)/n,
            "exact_wrong_rate": sum(r["left_exact_reward"] < r["right_exact_reward"] for r in sub)/n,
            "ai_better_rate": sum(r["preferred_variant"] == r["left_variant"] for r in sub)/n,
            "ai_tie_rate": sum(r["preferred_variant"] == "tie" for r in sub)/n,
            "ai_wrong_rate": sum(r["preferred_variant"] == r["right_variant"] for r in sub)/n})
    by = {r["comparison"]:r for r in rates}
    sensitivity = [{"feedback": name, "S_reason": by["reasoning_sensitivity"][f"{prefix}_better_rate"],
        "S_outcome": by["outcome_sensitivity"][f"{prefix}_better_rate"]}
        for name, prefix in [("exact_verifier", "exact"), ("ai_judge", "ai")]]
    table("task5_controlled_rates_verified", rates)
    table("task5_sensitivity_verified", sensitivity)
    pairwise, transfer = [], []
    summaries = {}
    for dataset, stem in [("GSM8K", "gsm"), ("SVAMP", "transfer")]:
        s = json.loads(repo_path(f"results/task5_feedback/{stem}_summary.json").read_text())
        summaries[dataset] = s
        for pair, v in s["pairwise"].items():
            left, right = pair.split("_vs_")
            n, counts = v["comparisons"], v["winner_counts"]
            pairwise.append({"dataset": dataset, "comparison": pair,
                "left_win_rate_ties_half": (counts.get(left,0)+.5*counts.get("tie",0))/n,
                "right_win_rate_ties_half": (counts.get(right,0)+.5*counts.get("tie",0))/n,
                "ties": counts.get("tie",0), "examples": n})
            v["left_win_rate_ties_half"] = pairwise[-1]["left_win_rate_ties_half"]
            v["right_win_rate_ties_half"] = pairwise[-1]["right_win_rate_ties_half"]
        save_json(f"results/task5_feedback/{stem}_summary.json", s)
    for policy in ("sft", "rlvr", "rlaif"):
        gsm = summaries["GSM8K"]["policies"][policy]["exact_accuracy"]
        svamp = summaries["SVAMP"]["policies"][policy]["exact_accuracy"]
        transfer.append({"policy": policy,"gsm_accuracy":gsm,"svamp_accuracy":svamp,
            "svamp_minus_gsm":svamp-gsm, "gsm_minus_svamp_drop":gsm-svamp})
    table("task5_pairwise_win_rates_verified", pairwise)
    table("task5_transfer_changes_verified", transfer)
    result = {"controlled_rates": rates, "sensitivity": sensitivity,
              "pairwise_win_rates": pairwise, "transfer_changes": transfer}
    save_json("results/task5_feedback/verified_analysis.json", result)
    controlled_path = repo_path("results/task5_feedback/controlled_diagnostics_summary.json")
    controlled = json.loads(controlled_path.read_text())
    controlled["controlled_preference_rates"] = rates
    controlled["strict_sensitivity"] = sensitivity
    save_json(controlled_path, controlled)
    combined_path = repo_path("results/task5_feedback/feedback_comparison_summary.json")
    combined = json.loads(combined_path.read_text())
    combined["controlled_diagnostics"] = controlled
    combined["gsm_pairwise"] = summaries["GSM8K"]["pairwise"]
    combined["transfer_pairwise"] = summaries["SVAMP"]["pairwise"]
    combined["verified_analysis"] = result
    save_json(combined_path, combined)
    return result


def task4_figures():
    rows = read_jsonl("results/task4_safety/judged_all.jsonl")
    policies = ["sft", "dpo", "ppo", "grpo"]
    categories = sorted({r["type"] for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    distributions = []
    for ax, policy in zip(axes.flat, policies):
        vals = np.zeros((len(categories), len(LABELS)))
        for i, cat in enumerate(categories):
            sub = [r for r in rows if r["policy"]==policy and r["type"]==cat]
            # Preserve the judge's actual labels, including its category errors.
            counts = Counter(r["ai_label"] for r in sub)
            for j, label in enumerate(LABELS):
                vals[i,j] = counts[label]/len(sub)
                distributions.append({"policy":policy,"category":cat,"label":label,
                    "count":counts[label],"category_examples":len(sub),"rate":vals[i,j]})
        ax.imshow(vals, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        for i in range(len(categories)):
            for j in range(5):
                if vals[i,j]: ax.text(j,i,f"{vals[i,j]:.0%}",ha="center",va="center",fontsize=7,
                                     color="white" if vals[i,j]>.55 else "black")
        ax.set_title(policy.upper())
        ax.set_yticks(range(len(categories)), categories, fontsize=8)
        ax.set_xticks(range(5), [x.replace("_", "\n") for x in LABELS], fontsize=7)
        ax.tick_params(labelbottom=True, labelleft=True)
    fig.suptitle("Actual AI-judge label distribution by XSTest category", fontsize=12)
    fig.tight_layout(rect=(0,0,1,.97))
    fig.savefig(FIG/"task4_category_label_distribution.png",dpi=300,bbox_inches="tight")
    plt.close(fig)
    table("task4_category_label_distribution_verified", distributions)
    return manual_audit_figure(rows)


def manual_audit_figure(rows):
    audit = list(csv.DictReader(repo_path("results/task4_safety/manual_audit_sheet.csv").open()))
    lookup = {(str(r["xstest_id"]),r["policy"]):r["ai_label"] for r in rows}
    matrix = np.zeros((5,5), dtype=int)
    pairs = []
    for r in audit:
        manual = r["manual_label"].strip().upper()
        if manual not in LABELS: raise ValueError("Incomplete or invalid human audit")
        ai = lookup[(str(r["xstest_id"]),r["policy"])]
        matrix[LABELS.index(manual),LABELS.index(ai)] += 1
        pairs.append((r["policy"],manual,ai))
    n = int(matrix.sum())
    sums = matrix.sum(1,keepdims=True)
    perc = np.divide(matrix, sums, out=np.zeros_like(matrix,dtype=float),where=sums>0)
    fig, ax = plt.subplots(figsize=(6,5.2))
    ax.imshow(perc,vmin=0,vmax=1,cmap="Blues")
    for i in range(5):
        for j in range(5):
            label = f"{matrix[i,j]}\n{perc[i,j]:.0%}" if sums[i,0] else "0\nN/A"
            ax.text(j,i,label,ha="center",va="center",fontsize=9,
                    color="white" if perc[i,j]>.55 else "black")
    labels = [x.replace("_","\n") for x in LABELS]
    ax.set_xticks(range(5),labels,fontsize=8)
    ax.set_yticks(range(5),labels,fontsize=8)
    ax.set_xlabel("AI judge label")
    ax.set_ylabel("Manual label")
    agreement = float(matrix.trace()/n)
    ax.set_title(f"Manual audit: agreement {agreement:.1%}, n={n}")
    fig.tight_layout()
    fig.savefig(FIG/"task4_manual_audit_confusion.png",dpi=300,bbox_inches="tight")
    plt.close(fig)
    result = {"labeled_examples":n,"overall_exact_agreement":agreement,
        "manual_ambiguous_rate":sum(m=="AMBIGUOUS" for _,m,a in pairs)/n,
        "ai_ambiguous_rate":sum(a=="AMBIGUOUS" for _,m,a in pairs)/n,
        "labels":LABELS,"confusion_manual_rows_ai_columns":matrix.tolist(),
        "figure_label_semantics":"Actual unmodified ai_label, not benchmark-remapped behavior labels."}
    provenance = repo_path("results/task4_safety/audit_correction_provenance.json")
    if provenance.exists():
        result["review_type"] = json.loads(provenance.read_text())["review_type"]
        result["provenance_file"] = "results/task4_safety/audit_correction_provenance.json"
    save_json("results/task4_safety/verified_audit_analysis.json",result)
    return result


def handoff(summary):
    lines = ["# Verified experiment data and completion notes", "",
        "Original training outputs are preserved. Task 1 completion diagnostics are a new frozen-checkpoint evaluation. No model weights were updated.", "",
        "## Corrections to earlier handoffs", "",
        "- Task 5 strict reasoning sensitivity: exact verifier 0.00; AI judge 0.10. Agreement with a tied verifier is not reasoning sensitivity.",
        "- Exact verifier filler comparison: better 0.00, tie 1.00, wrong 0.00.",
        "- Use sample standard deviation for consistent response-length dispersion. Earlier IQR values used a different order-statistic convention.",
        "- Task 3 group-size analysis partitions all eight cached completions of each of the same 24 prompts. There are 96/48/24 groups for K=2/4/8, not 96/48/24 distinct prompts.",
        "- The Task 1 Chinese/French dataset example does not establish that the chosen answer is worse than the rejected answer; both are defective. Do not call dataset responses final-policy generations.", ""]

    def section(title, rows):
        lines.extend([f"## {title}", ""])
        if not rows:
            lines.extend(["No completed data available.", ""])
            return
        keys = list(rows[0])
        lines.extend(["| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys)+" |"])
        for row in rows:
            vals = [f"{row[k]:.8g}" if isinstance(row[k],float) else str(row[k]) for k in keys]
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")
    dpo = []
    for path in sorted(repo_path("results/task1_dpo").glob("*/eval_summary.json")):
        s = json.loads(path.read_text())
        dpo.append({"run":path.parent.name,"pair_examples":s['preference']['examples'],
            "accuracy":s['preference']['preference_accuracy'],"loss":s['preference']['loss'],
            **{k:s['generation'][k] for k in ['reward_model_score','sampled_kl','mean_response_length']}})
    section("Task 1 original saved evaluation",dpo)
    if len(summary["task1"].get("runs",{})) == 2:
        with (FIG/"task1_strata_and_word_limits_verified.csv").open() as f:
            section("Task 1 additional frozen-checkpoint diagnostics",list(csv.DictReader(f)))
    else:
        lines.extend(["## Task 1 additional diagnostics", "", "Still running; check results/task1_dpo/completion_diagnostics/summary.json.", ""])
    for task in ("task2_ppo", "task3_grpo"):
        rows=[]
        for path in sorted(repo_path(f"results/{task}").glob("*/eval_summary.json")):
            s=json.loads(path.read_text())
            rows.append({"run":path.parent.name,**{k:s[k] for k in ['examples','reward_model_score','sampled_kl','entropy','mean_response_length','terminated_rate']}})
        section(task + " original held-out metrics",rows)
    section("Task 2 stability (pre-clipping policy gradient norm)",summary['saved_metrics']['ppo_stability'])
    section("Task 3 post-hoc log-probability gradient diagnostic",summary['grpo']['rows'])
    with (FIG/"task3_group_size_table.csv").open() as f:
        section("Task 3 group-size and difficulty-bin diagnostics",list(csv.DictReader(f)))
    lines.extend([summary['grpo']['scope'],"",summary['grpo']['method'],""])
    safety=json.loads(repo_path('results/task4_safety/summary.json').read_text())
    section("Task 4 policy metrics",[{"policy":p,**{k:v[k] for k in ['safe_answer_rate','over_refusal_rate','justified_refusal_rate','unsafe_compliance_rate','ambiguous_rate','mean_response_tokens']}} for p,v in safety['policies'].items()])
    section("Task 4 raw judge/manual agreement",[{k:summary['task4'][k] for k in ['labeled_examples','overall_exact_agreement','manual_ambiguous_rate','ai_ambiguous_rate']}])
    section("Task 5 controlled rates",summary['task5']['controlled_rates'])
    for dataset, stem in [("GSM8K","gsm"),("SVAMP","transfer")]:
        data=json.loads(repo_path(f"results/task5_feedback/{stem}_summary.json").read_text())
        section(f"Task 5 {dataset} policy metrics",[{"policy":p,**{k:m[k] for k in ['examples','exact_accuracy','designated_final_rate','mean_response_tokens','truncated_rate']}} for p,m in data['policies'].items()])
    section("Task 5 strict sensitivities",summary['task5']['sensitivity'])
    section("Task 5 pairwise rates with ties worth one half",summary['task5']['pairwise_win_rates'])
    section("Task 5 transfer exact accuracy",summary['task5']['transfer_changes'])
    section("Response-length dispersion from original saved generations",summary['saved_metrics']['response_length_dispersion'])
    lines.extend(["## Sources and interpretation limits", "",
        "- Raw judge labels are preserved in the new Task 4 category-distribution figure. These differ from benchmark-remapped safety outcome rates.",
        "- Task 2 raw and effective terminal rewards with exact response text: results/completion_audit/exact_scored_ppo_cache_examples.json. These are supplied midpoint cache examples, not final-policy generations.",
        "- Selected FINAL-policy responses now have new post-hoc reward scores: results/completion_audit/scored_final_policy_examples.jsonl. Exact comparison texts and assessment notes: report/QUALITATIVE_EVIDENCE_VERIFIED.md.",
        "- Task 5 exact controlled texts: results/completion_audit/exact_task5_diagnostic_examples.json.",
        "- The original final-policy generation files omit per-response reward scores. Do not attach a run-average reward to any individual response.",
        "- Task 1 new preference margins are in completion_diagnostics/*_preference_examples.jsonl; word-limit responses, counts and compliance are in *_word_limit_generations.jsonl.",
        "- Tiny numerical changes in a later frozen evaluation may change near-zero DPO margins. Keep the original aggregate metrics and label new stratified measurements as post-hoc evaluation.",
        "- Standard PPO continuation time: 379.0508 seconds. Standard GRPO: 1102.2309 seconds. These exclude initial model loading and separate evaluation.",
        "- Historical peak MPS memory was not recorded. The CUDA-only zero placeholder is not measured memory usage and cannot be recovered exactly after the run.",
        "- Logged entropy is mean negative sampled-token log probability, not an exact full-vocabulary entropy calculation.",
        "- Cached PPO clipping compares reference versus old probabilities; distinguish it from updated-policy clipping during training.",
        "- The greedy generation fallback is not logged. Whether it triggered in original runs cannot be reconstructed.",
        "- Fixed update counts and generation caps do not imply exactly equal realized generated-token counts.",
        "- No Git repository exists in this workspace; no commit or push was performed.", "",
        "## Reproduction commands", "", "```bash",
        "python -m scripts.complete_task1_evidence  # frozen evaluation only; resumable",
        "python -m scripts.make_report_figures      # saved results only",
        "python -m unittest discover -s tests -v",
        "python -m scripts.validate_assets", "```", "",
        "Observed dependency versions: results/completion_audit/observed_environment.json. Course pins are kept in requirements.txt; the observed environment differs and a clean install of those pins was not tested during this audit."])
    lines += ["", "## Exact configuration files", ""]
    for name in ['base','dpo','ppo','grpo','feedback']:
        lines += [f"### configs/{name}.yaml", "", "```yaml",
                  repo_path(f"configs/{name}.yaml").read_text().rstrip(), "```", ""]
    repo_path("report/VERIFIED_DATA_HANDOFF.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notes", action="store_true", help="Also export optional report-helper documents")
    args = parser.parse_args()
    FIG.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":9,"axes.titlesize":10,"savefig.facecolor":"white"})
    summary = {"saved_metrics":saved_metrics(), "grpo":grpo_normalization(),
        "task5":task5(), "task4":task4_figures(), "task1":task1_tables(args.notes),
        "qualitative_sources":qualitative_sources(args.notes)}
    save_json(OUT/"saved_evidence_summary.json",summary)
    # Store the actual local versions without rewriting the course dependency pins.
    packages = ["torch", "transformers", "tokenizers", "trl", "peft", "datasets",
                "accelerate", "huggingface_hub", "safetensors", "pyyaml", "pandas",
                "numpy", "scikit-learn", "matplotlib", "tqdm"]
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    save_json(OUT / "observed_environment.json", versions)
    if args.notes:
        handoff(summary)
    # Refresh the evidence inventory after all derived files exist.
    from scripts.make_report_figures import write_evidence_status
    write_evidence_status()
    print(f"Saved diagnostics and figures: {OUT / 'saved_evidence_summary.json'}")


if __name__ == "__main__":
    main()
