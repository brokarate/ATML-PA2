# ATML PA2 - LLM Post-Training

<!-- FINAL_STUDENT_SETUP -->

## Quick start

```bash
git clone https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining.git
cd ATML-PA2-LLM-PostTraining
python -m pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
```

The fixed datasets, cached diagnostics, and supplied
continuation checkpoints are downloaded from:

https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets

Pinned release revision:

`0b350481fb03f5525a35bcdec4131bd4fe487f98`

## Student run notes

These experiments were run on a Mac using Apple MPS instead of the CUDA setup used by the release authors. The code keeps the assignment settings and hyperparameters the same, but includes small device/numerical-stability fixes so PPO/GRPO can run reliably on this machine. In particular, reward/judge models may run unquantized on Mac because the released quantized setting is CUDA-specific.

For Task 1, a small number of examples whose prompts were already too long for the configured sequence limit were skipped and logged. This does not change the dataset files themselves; it only prevents those overlong examples from crashing training/evaluation on this local setup.

## Final evidence completion (no training)

```bash
python -m scripts.complete_task1_evidence
python -m scripts.make_report_figures
python -m unittest discover -s tests -v
```

The first command performs resumable frozen-checkpoint preference evaluation by
length stratum and generates responses to the fixed ten word-limit prompts.
It saves new measurements under `results/task1_dpo/completion_diagnostics/`
without replacing the original aggregate evaluations or model checkpoints.
The plotting command also runs `scripts.complete_saved_evidence`, which uses
saved data only to rebuild the Task 4 category/audit figures, compute PPO
stability and response-length dispersion, and verify Task 5 strict sensitivity,
tie-aware pairwise rates, and transfer changes. All raw judge decisions remain
unchanged. Machine-readable diagnostics are saved in
`results/completion_audit/saved_evidence_summary.json`; report PNGs and compact
CSV tables are saved in `report/figures/`. Optional report-helper documents
are exported only when explicitly running `python -m scripts.complete_saved_evidence --notes`.

The GRPO length-conditioned diagnostic differentiates the policy surrogate with
respect to token log probabilities on the fixed cache regrouped into K=4, with the same
advantages and 512-token truncation mask for both normalizations. It is a
post-hoc normalization mechanism check, not a historical parameter-gradient
measurement from the K=4 training forks. Its scope is recorded in the saved JSON.

The installed package versions are recorded in
`results/completion_audit/observed_environment.json`. They differ from the
original course pins in `requirements.txt`; those pins have not been reinstalled
and tested during this audit. `requirements-mps-observed.txt` records the direct
versions used locally, without claiming a tested clean-install lockfile.
On MPS, CUDA-specific frozen-model quantization
is disabled. Generated inference tensors are cloned before PPO autograd reuse;
log-softmax uses float32 and AdamW epsilon is 1e-5. Nonfinite losses/gradients
are guarded and skipped updates are logged. The generation helper can retry an
invalid sampling distribution with greedy decoding, but original runs did not
record whether this fallback was used. Historical peak MPS memory was not
measured; `peak_vram_bytes=0` is a CUDA-only placeholder, not a memory result.

In the evidence tables, a sampled entropy diagnostic means negative mean
sampled-token log probability. The cached PPO clipping diagnostic uses the
reference/old probability ratio. Equal update counts and generation caps do not
guarantee identical realized token counts. These distinctions must be preserved
when interpreting the saved experiments.

For individually scored qualitative examples without regenerating any response,
run `python -m scripts.score_saved_qualitative` with MPS available. It scores a
small, explicitly selected set of original saved responses and records these
as new post-hoc measurements in
`results/completion_audit/scored_final_policy_examples.jsonl`.

---
# ATML PA2 - LLM Post-Training

This is the **student starter repository** for ATML PA2. The released code is intentionally incomplete: Tasks 1-3 provide model/data loading, objective helpers, checkpoint restoration, and experiment entry points, but **you must implement the training loops and ablation orchestration yourself**. Each of Tasks 1-3 also contains one deliberate algorithmic defect in its core objective code; identifying and correcting these defects is part of validating your implementation.

Task 4 supplies the fixed AI safety judge and response-generation utilities, but you must write the evaluation/aggregation code. Task 5 supplies the exact RLVR verifier, the fixed pairwise AI judge used for RLAIF evaluation, and data/model loaders; you must implement the requested evaluation and analysis.

## 1. Clone and install

```bash
git clone https://github.com/COURSE_ORG/ATML-PA2-LLM-PostTraining.git
cd ATML-PA2-LLM-PostTraining
python -m pip install -r requirements.txt
```

## 2. Download the course assets

The large course-created checkpoints and fixed data are distributed as a GitHub Release asset rather than normal Git files. After cloning, run:

```bash
python -m scripts.download_assets
python -m scripts.validate_assets
```

If your instructor provides a direct asset URL separately, use:

```bash
python -m scripts.download_assets --url '<ASSET_URL>'
```

Public base/reward/judge models are downloaded from Hugging Face at runtime and are **not** included in the course asset archive.

The installer also materializes the fixed 100-example Task 5 transfer set from the official SVAMP challenge-set source if it is not already present. The tiny Task 1 word-limit prompt set is tracked directly in this repository.

## 3. Environment check

```bash
python -m scripts.check_environment
```

Run commands from the repository root. The reference environment used to prepare the release pins Transformers 4.57.1, TRL 0.27.2, PEFT 0.17.1, and Tokenizers 0.22.1.

## 4. Supplied course checkpoints

After `download_assets`, these directories should exist:

```text
checkpoints/ppo_midpoint_policy/
checkpoints/ppo_midpoint_value/
checkpoints/grpo_midpoint_policy/
checkpoints/rlvr_policy/
checkpoints/rlaif_policy/
```

PPO and GRPO begin from the supplied continuation checkpoints. RLVR and RLAIF are supplied frozen evaluation policies; students do not retrain them.

The PPO value checkpoint is intentionally released as the exact staff midpoint state, including its imperfect held-out value calibration. Treat critic behavior as an analysis variable rather than assuming a perfect baseline, and start every PPO fork from the identical supplied policy/value state. The default continuation generation cap is 512 tokens for feasibility; frozen evaluation uses the larger cap specified in `configs/ppo.yaml`.

## 5. Task entry points

### Task 1 - DPO

```bash
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.ablate_beta --config configs/dpo.yaml
python -m task1_dpo.analyze_length --config configs/dpo.yaml
```

### Task 2 - PPO

```bash
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml
python -m task2_ppo.ablate_kl --config configs/ppo.yaml
```

### Task 3 - GRPO

```bash
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml
python -m task3_grpo.compare_normalization --config configs/grpo.yaml
```

### Task 4 - Safety calibration

The judge loader/parser are supplied. You must implement the requested generation aggregation and evaluation.

```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml
python -m task4_safety.judge_responses --config configs/feedback.yaml
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml
python -m task4_safety.evaluate_safety --config configs/feedback.yaml
```

### Task 5 - RLVR vs RLAIF

The exact verifier and pairwise AI judge are supplied; you implement the evaluation/analysis.

```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.score_perturbations --config configs/feedback.yaml
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
```

## 6. Reproducibility rules

- Do not alter course-provided data, cached rollouts, or supplied checkpoints.
- Start every short fork from the **same supplied midpoint checkpoint**.
- Keep prompt IDs, generated-token/update budgets, seed, and evaluation procedure matched across ablations.
- Commit your code, configs, small JSON/CSV logs, and figures. Do not commit downloaded checkpoints, raw course assets, or model caches.
- Record peak VRAM and wall-clock time for the standard PPO and GRPO continuations.

See the assignment manual for the required experiments, metrics, and report questions.
