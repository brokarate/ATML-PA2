Task 5 notes
============

Task 5 evaluates the supplied frozen RLVR and RLAIF policies; it does not retrain them.

The implementation writes resumable outputs under `results/task5_feedback/`. If a run is
interrupted, rerun the same command and completed generations or judge comparisons will be
reused where possible.

On Mac/MPS, the tokenizer and pairwise judge are loaded with the same assignment models and
rubrics, but with local-cache-first loading and unquantized device placement. This avoids
CUDA-only quantization assumptions and Hugging Face metadata lookups during long local runs.
