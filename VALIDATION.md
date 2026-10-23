# First milestone verification

Verified on 23 October 2026 with Python 3.12.14 on Windows using the accompanying dependency lockfile.

- Automated suite: 21 passed, zero failures or skips. Covers split isolation, outer-test independence, fold-local preprocessing, unknown categories, capped feature selection, exact budgets, reproducibility, actual PPO execution, Gymnasium's environment checker, JSON trajectory round-trip, cold-start arithmetic, and terminal potential telescoping.
- `results/verified-pilot`: three generated tasks, three methods, eight evaluations each; 72 pipeline evaluations, zero failed candidates. Each method uses the same task split, inner folds, search menu, and seed.
- `artifacts/verified-bank`: three generated meta-training tasks, eight PPO episodes each; 24 complete trajectories, 96 logged transitions, zero failed candidates. No outer-test scores were computed for bank generation.
- Each default PPO run collects two 16-transition rollouts, updating the policy between them. Three policy checkpoints are included in each pilot/bank run.

These are smoke checks of execution and contracts. They do not establish that PPO learns an effective search policy or that trajectory transfer improves search. Timing includes startup costs, and some smoke checks overlapped with tests; do not use these timings for scientific method comparisons. Earlier `initial-*` local runs remain as development history; the packaged project includes only the verified runs.

The next substantive milestone is an independently licence-checked real dataset inventory and a costed three-dataset pilot, followed by a training-only bank large enough to evaluate the LSTM encoder. Learned retrieval, transfer warm-starts, shaped PPO, repeated-seed statistical analysis, a final meta-test runner, and a dashboard are not implemented in this milestone. The `shaping.py` module contains tested arithmetic only.
