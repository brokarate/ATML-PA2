# PA2 Evidence Artifacts

Generated from saved experiment outputs only. No model training, model loading, or new evaluation is performed by this plotting script.

- Task 1: DPO summary: beta/loss/preference accuracy/KL/reward/length -> task1_dpo_summary_table.csv; task1_dpo_beta_ablation.png (covered from saved results)
- Task 1: Length-confounding diagnostics -> task1_strata_and_word_limits_verified.csv; task1_length_balanced_eval_table.csv (covered by separate frozen-checkpoint evaluation)
- Task 1: Qualitative examples -> not generated as prose (requires manual selection/interpretation from saved generations)
- Task 2: PPO continuation trajectories -> task2_ppo_standard_trajectory.png (covered from train_log)
- Task 2: Clipping and KL ablations -> task2_ppo_ablations.png; task2_clipping_ablation_table.csv; task2_kl_ablation_table.csv; task2_stability_statistics.csv (covered from saved summaries)
- Task 3: GRPO continuation trajectories -> task3_grpo_standard_trajectory.png (covered from train_log)
- Task 3: Group size and normalization comparison -> task3_group_size_and_normalization.png; task3_group_size_table.csv; task3_normalization_eval_table.csv; task3_length_conditioned_normalization.csv (held-out fork metrics plus post-hoc K=4 regrouped-cache log-probability-gradient diagnostic; not historical parameter gradients)
- Task 4: Safety calibration by policy -> task4_safety_policy_summary.png; task4_safety_summary_table.csv (covered from judged outputs)
- Task 4: Category-level XSTest behavior -> task4_category_heatmaps.png; task4_category_metrics_table.csv; task4_category_label_distribution.png (covered from category metrics)
- Task 4: Manual audit agreement -> task4_manual_audit_confusion.png; results/task4_safety/verified_audit_analysis.json (240 labels complete; exact raw-label agreement 0.720833; includes human-approved AI-assisted post-hoc corrections)
- Task 5: GSM/SVAMP policy comparison and pairwise judge -> task5_gsm_svamp_policy_comparison.png; task5_policy_metrics_table.csv; task5_pairwise_judge_table.csv (covered from saved results)
- Task 5: Controlled perturbation diagnostics -> task5_controlled_feedback_diagnostics.png; task5_controlled_pairwise_rates.csv; task5_sensitivity_verified.csv (covered from saved diagnostics)
- Task 2/3: Peak memory -> standard/summary.json (MPS memory was not recorded; zero is a CUDA-only placeholder, not a measurement)
