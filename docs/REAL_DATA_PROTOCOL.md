# Real dataset pilot protocol

This milestone tests data admission, repeatable baseline execution, and real PPO trajectory generation. It does not test trajectory transfer, prove PPO superiority, or complete the full benchmark selection.

## Scope and admission

The candidate inventory contains 40 UCI datasets identified by official source IDs. Candidate metadata is not admission evidence: entries remain pending, deferred, or excluded unless their individual source page, role schema, and data pass the pilot audit. OpenML mirror discovery and a broader related-version audit remain necessary before freezing the full research pool.

Eight datasets are admitted to pilot-v1. Wine, Breast Cancer Wisconsin Diagnostic, and Ionosphere are meta-training tasks. Glass Identification and Banknote Authentication are reserved for validation. Iris, German Credit, and Sonar are reserved for meta-test. All assignments were made before model runs, using source/size/schema considerations, not performance. This is a small pilot with limited task diversity, not a domain-disjoint split or a sufficient meta-learning benchmark.

Each admitted dataset has an individually checked UCI page declaring CC BY 4.0. The audit records its canonical source and data URLs, DOI, creators, access timestamp, page hash, metadata hash, raw checksum, processed checksum, feature schema, target, class counts, missingness, excluded IDs, duplicate-row count, and a review note. The source dataset ID and any source row identifier are not model features.

Raw and processed datasets remain local. Recreate processed files with:

```powershell
.\.venv\Scripts\python.exe -m trajmeta.benchmark restore --output data/audits/pilot-v1
```

Restore verifies the frozen audit files and all source/processed checksums. If upstream metadata or data changes, it stops for review instead of replacing the frozen version. Creating a new inventory or audit requires a separate output path:

```powershell
.\.venv\Scripts\python.exe -m trajmeta.benchmark inventory --output data/candidates-next.csv
.\.venv\Scripts\python.exe -m trajmeta.benchmark audit --output data/audits/pilot-next
```

The audit command uses the explicitly reviewed selection in `benchmark.py`; it does not automatically admit the entire candidate inventory. The existing generated-data configuration remains separate for regression tests.

## Data quality and leakage controls

Only documented Feature columns and the single Target column are retained. Numeric-coded categorical features retain their categorical role on reload. The raw data is preserved; the processed CSV drops exact duplicate rows before any row split. Identical features with contradictory targets are rejected until group-aware splitting is implemented. Canonical row fingerprints detect exact equal-content datasets independent of row order. Related dataset versions still require source review: this does not detect arbitrary rescaling, renamed/reordered feature variants, or all hidden subject relationships.

The first audit removed 1 duplicate row from Ionosphere, 1 from Glass, 24 from Banknote Authentication, and 3 from Iris. Breast Cancer's `ID` and Glass's `Id_number` were excluded. Vehicle Silhouettes was deferred after its canonical CSV contained a singleton `204` class in addition to bus/opel/saab/van. No model score influenced that decision and no source label was silently repaired.

The pilot policy caps admitted data at 5,000 rows, 60 input features, and 20 MB per download. Trials execute in a single spawned worker with a 120-second deadline. Timed-out workers are terminated; timeout, worker-exit, and caught memory errors consume the trial budget and are logged. This is not a hard RAM cap. Audit limitations include unobserved repeated subjects, high-cardinality encoding expansion, and source documentation errors.

Dataset-level audit may inspect target validity and class counts for split feasibility; it never fits a model on reserved tasks. The development CLI rejects meta-test tasks, checks frozen manifest hashes, and verifies processed CSV checksums before loading. These are workflow safeguards, not an operating-system access-control boundary.

## Fixed experiment design

`configs/real_smoke.yaml` uses one seed and eight evaluations per method on three meta-training tasks: 72 total evaluations. `configs/real_pilot.yaml` uses algorithm seeds 11, 29, and 47 and 32 evaluations per method: 864 total evaluations. Both use a fixed row-split seed of 2026, a 25% outer holdout, and identical two-fold inner stratified CV across methods and algorithm seeds. Two folds are a pilot compute choice, not a tuned setting.

The same 48 configurations and macro-F1 objective apply to Random Search, Optuna, and PPO. Failed and repeated configurations consume budget. Each full-pilot PPO run has four complete rollout updates. Policy optimization and classifier fitting are separate processes; no retrieval or cross-task policy transfer is enabled.

Outer-test reporting is disabled for this development pilot. No validation-derived quality target has been selected, so N_target remains unset. Do not derive N90 from a method's own best result or use pilot test outcomes to define it. Final generalization experiments and validation-selected targets remain future work.

Run the cost pilot first, then the repeated pilot:

```powershell
.\.venv\Scripts\python.exe -m trajmeta.repeat --config configs/real_smoke.yaml --output results/real-smoke-v1
.\.venv\Scripts\python.exe -m trajmeta.repeat --config configs/real_pilot.yaml --output results/real-pilot-v1
```

The repeat command verifies a signature of code, configuration, and manifests. Re-running the same command skips completed dataset/method/seed units. Every evaluation is appended to a JSONL log. An interrupted unit is retained and requires inspection; it is not silently replayed or counted as a free retry. PPO cannot resume mid-rollout in this milestone. Source changes require a new output directory.

PPO episodes are four-step pipeline-construction trajectories, not multi-candidate search histories. The future encoder must use that definition consistently. A separate experiment would be needed to claim benefit from transferring the ordering of many candidate evaluations.

Build the bank from already completed training runs without extra evaluations:

```powershell
.\.venv\Scripts\python.exe -m trajmeta.bank_from_runs --runs results/real-pilot-v1 --output artifacts/real-bank-v1/trajectories.jsonl
```

All episodes, including failed trials if any occur, are retained with status. Encoder supervision must distinguish failures from low-quality successful trials and use within-dataset ranks. Split complete trajectories into encoder train/check partitions before generating prefixes so one trajectory cannot leak across those partitions.

## Gates still open

The initial eight-task admission gate can pass while the full research Gate A remains open. More independent training tasks, explicit OpenML mirror checks, and broader group/temporal leakage review are required. PPO learning quality, embedding quality, retrieval utility, shaping benefit, and generalization remain unproven. No LSTM, dashboard, or additional search space was added in this milestone.
