# Task 5 Feedback Comparison

## In-domain GSM8K
- rlaif: exact accuracy 0.297, designated-final rate 0.407, mean response tokens 289.1
- rlvr: exact accuracy 0.290, designated-final rate 0.420, mean response tokens 283.9
- sft: exact accuracy 0.293, designated-final rate 0.400, mean response tokens 284.1

## Transfer Math
- rlaif: exact accuracy 0.570, designated-final rate 0.630, mean response tokens 171.2
- rlvr: exact accuracy 0.450, designated-final rate 0.570, mean response tokens 168.1
- sft: exact accuracy 0.560, designated-final rate 0.670, mean response tokens 162.5

## Controlled Diagnostics
- distractor_robustness: judge/exact agreement 0.000; preferences {'tie': 20}
- filler_susceptibility: judge/exact agreement 0.350; preferences {'persuasive_filler_correct': 11, 'tie': 7, 'clean_correct': 2}
- final_vs_reasoning: judge/exact agreement 0.200; preferences {'tie': 16, 'corrupt_reasoning_correct_final': 4}
- outcome_sensitivity: judge/exact agreement 0.150; preferences {'tie': 17, 'clean_correct': 3}
- reasoning_sensitivity: judge/exact agreement 0.600; preferences {'clean_correct': 2, 'tie': 12, 'corrupt_reasoning_correct_final': 6}

## Notes
RLVR is scored by the exact designated-final verifier. RLAIF preferences are scored by the fixed pairwise AI judge without access to gold answers. Interpret judge-based comparisons alongside exact accuracy and the controlled perturbation diagnostics.
