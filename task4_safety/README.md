# Task 4 Safety Calibration

Run this only after Task 3 has produced `outputs/task3_grpo/standard`.

```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml
python -m task4_safety.judge_responses --config configs/feedback.yaml
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml
python -m task4_safety.evaluate_safety --config configs/feedback.yaml
```

The scripts keep the fixed XSTest prompt order and save one generated-response file per policy plus combined files under `results/task4_safety/`.

Mac/MPS note: the same assignment models and config values are used, but frozen judge/model loading may run unquantized on Mac because the released 4-bit quantized path is CUDA-specific.
