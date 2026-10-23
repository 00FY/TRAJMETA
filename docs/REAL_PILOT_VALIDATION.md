# Real data pilot verification

The initial real-data milestone completed 27 runs and 864 candidate evaluations on three meta-training datasets, using three algorithms and three seeds. Every run used 32 evaluations. All trials completed successfully; 155 trials recorded model or feature-selection warnings, retained in the raw evaluation evidence. Warnings are not suppressed or represented as failures.

The sum of recorded search time was 128.07 seconds (2.13 minutes), including controller setup, CV worker startup, and PPO checkpoint writing. This is measured pilot cost, not a projection of full-benchmark compute. The initial 72-evaluation cost smoke test was additional work and is preserved locally in results/real-smoke-v1.

The training bank contains 288 complete pipeline-construction trajectories and 1,152 transitions, balanced across the three tasks and three seeds. All nine PPO runs contributed 32 trajectories each. Bank quality summaries show the within-task score range and number of distinct outcomes; they are not encoder sanity checks. The LSTM has not been trained.

No outer-test score was computed. No model was run on meta-validation or meta-test datasets. There is no selected N_target threshold and no claim of transfer benefit, PPO superiority, or a 30% improvement. Completed-run resume was exercised and skipped all 27 units without further evaluations.

The committed evidence includes plan.json, summary.json, evaluations.jsonl (including anytime best-inner scores), verification.json, bank_quality.json, and a bank checksum manifest. Full checkpoints, original datasets, and full trajectory arrays are stored locally and can be regenerated using docs/REAL_DATA_PROTOCOL.md. The source hashes in local provenance records identify the run implementation; their Git commit field refers to the parent commit because this code was tested before it was committed.

The full regression suite passed 32 tests. The 155 warning-bearing trials comprise 124 constant-feature selection cases (with undefined univariate F statistics handled by scikit-learn) and 31 logistic-regression convergence-limit cases. These produced finite scores under the fixed candidate definitions; they must not be described as fully converged fits. Review the iteration cap and feature-selector handling before scaling, using training/validation evidence only.
