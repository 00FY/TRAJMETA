# TrajMeta-AutoML

Trajectory-guided AutoML for tabular classification, with a shared LSTM, frozen two-stage retrieval, reward-shaped PPO, and controlled baseline comparisons.

**Status: software implemented and bounded real-data study completed.** The study audits 40 UCI candidates, admits 16 tasks, trains on 11, selects settings on two, and evaluates nine methods on three held-out tasks with three seeds. This small experiment does not establish broad transfer benefit or the proposed 30% improvement.

Read the [research report](reports/study-v1/REPORT.md), [study protocol](docs/TRANSFER_STUDY.md), [machine-readable statistics](reports/study-v1/statistics.json), and [dataset attribution](docs/DATA_ATTRIBUTION.md). The original synthetic and real-data pilots remain preserved.

## Open the connected UI

Run `.\scripts\run_ui.ps1` or double-click **Start TrajMeta.cmd**, then open **http://127.0.0.1:8765**. The local dashboard connects real searches, trajectory inspection, dataset provenance, and study comparisons. See the [UI guide](docs/WEB_APP.md) for setup, uploads, and operating limits.

## Verify or reproduce the transfer study

After environment setup below:

```powershell
.\scripts\run_study.ps1
```

This restores the published evidence and any missing audited datasets, runs tests, verifies saved runs, and regenerates the report. It does not count verification as a new experiment.

For an independent rerun after data restoration:

```powershell
.\.venv\Scripts\python.exe scripts/reproduce_study.py --scope full --output results/replication-001
```

Use `--scope final` to reuse the published frozen encoder/bank and rerun only the final comparisons. Replications use a new directory and report score differences without overwriting the original evidence. `--prepare-only` validates setup without starting model evaluations; `--resume` continues a matching replication.

For a small synthetic demonstration:

```powershell
.\.venv\Scripts\python.exe scripts/demo_transfer.py --budget 4
```

The frozen checkpoint and compressed training bank are under `models/study-v1`. Exact run records are in `reports/study-v1/experiment_evidence.zip`; raw data and per-run PPO policies remain local. Source/audit files preserve exact bytes in Git so provenance hashes survive checkout. The [documented metadata amendment](reports/study-v1/protocol_amendment.json) preserves all earlier results and original protocol evidence.

## Quick start

Python 3.11 or 3.12 is recommended. From this project folder in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m trajmeta.cli audit
.\.venv\Scripts\python.exe -m trajmeta.cli run --output results/my-pilot
.\.venv\Scripts\python.exe -m trajmeta.cli bank --output artifacts/my-bank
```

The lockfile records the tested Windows/Python 3.12 environment. If a different platform cannot resolve it, install `-e ".[dev,report]"`, run the tests, and freeze a separate lockfile. No execution-policy change is required. The project is stored at `D:\Desktop\TRAJMETA` and backed up at https://github.com/00FY/TRAJMETA. After setup, `scripts/run_local.ps1` runs the pilot using the project-local `.venv`. Per-run PPO policies stay local. Exact recorded runs, study results, the shared encoder and compressed training bank are versioned for inspection.

Output directories must be new. Runs never overwrite prior evidence. Default directories have unique UTC timestamps. `configs/pilot.yaml` selects three generated training tasks, eight pipeline evaluations per method, three inner folds, and one seed. PPO uses small networks and short updates for a smoke test, not a tuned research baseline.

## What works

- Manifest audit: unique dataset IDs, frozen split membership, duplicate groups, checksums/source duplication across splits, and provenance fields.
- Generated mixed numeric/categorical data with missing numeric values; audited local CSVs with SHA256 verification.
- Stratified outer train/test split; fixed inner folds shared across methods.
- Forty-eight pipelines combining three scalers, two selection modes, four classifiers, and two hyperparameter levels.
- Random Search, Optuna TPE over the same finite configuration menu, and Stable-Baselines3 PPO.
- Four-stage Gymnasium episodes with one evaluation per episode; terminal macro-F1 reward and zero intermediate reward.
- Versioned complete trajectory records, model checkpoints, per-trial outcomes, timing, package versions, source hash, and config/manifest snapshots.
- Training-only bank admission, shared LSTM quality supervision, metadata/trajectory retrieval, potential shaping, transfer ablations, protocol locks and cold-start parity checks.

## Experimental contract

The evaluator only receives outer-training rows. Imputation, categorical encoding, scaling, and selection are fitted separately in every inner training fold. When reporting is enabled, outer-test scores are computed once for the best successful candidate after each baseline search. Real-data development configurations disable this reporting. Bank generation never evaluates outer-test scores. No outer score is an observation or reward.

All methods consume exactly the configured number of evaluations, including repeat configurations and failed candidates. No method gets free cached scores. Final fitting is additional work outside the search budget. The summary's `search_seconds` includes controller overhead and PPO checkpoint writing; per-evaluation times are also logged. Runs are sequential CPU experiments. Real-data configurations isolate CV in a spawned worker with a configured worker response timeout; failed and timed-out trials count toward the budget. Three recorded trials exceeded that timeout in total elapsed time, so a strict end-to-end wall-time bound and wall-clock speed comparisons are not established; see the timing audit. There is no hard RAM limit, so use the audited dataset size bounds.

PPO receives a flat discrete action space whose size is the least common multiple of stage menu lengths. At each stage, action modulo menu length determines the choice. Every valid choice has the same number of aliases. The default has twelve action IDs and 48 complete configurations. The observation combines seven bounded training-data descriptors, one-hot partial choices, a stage indicator, the previous action, used-budget fraction, and best inner score. Dataset IDs are excluded. Dataset descriptors use fixed transforms, not statistics fitted on held-out datasets. The exact order is in `env.py`.

PPO rollout sizes divide the total transition budget exactly. Reset starts a new construction episode but does not replenish the evaluator's run budget. After the budget is exhausted, further evaluations raise an error. This guards against the full-rollout rounding behavior documented by [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html). The environment uses the [Gymnasium reset/step API](https://gymnasium.farama.org/introduction/create_custom_env/).

`target_score` is null by default, so no N90 or N_target claim is made. Set a fixed validation-derived target only after preregistration. N_target counts successful inner validation scores reaching that threshold; unreached targets remain null and must be treated as censored, not zero.

## Real data onboarding

Add one row per audited dataset to `data/dataset_manifest.csv` and exactly one matching split row to `data/split_manifest.csv`. Local CSV paths are relative to the project root. Record the original source URL/ID, verified licence, access date, target, domain, duplicate group, and SHA256 of the exact CSV. Set `verified=true` only after independently checking the individual source's reuse terms. The structural audit cannot establish legal permission or discover semantic near-duplicates for you. Shared source URLs are conservatively grouped; use dataset-specific canonical URLs.

The loader checks the checksum before loading. Missing target labels and insufficient class counts are rejected. CSV feature dtypes are inferred by pandas: inspect IDs, numeric-coded categories, date fields, and target leakage manually before admitting a real dataset. The generated fixtures are explicitly marked `generated`; their recipe is fixed in source and tracked by the source hash. They are not UCI/OpenML benchmark entries.

## Evidence and remaining research scope

The implemented study includes warning review, a frozen dataset audit, training-only trajectory collection, encoder sanity checks, validation-only selection, nine matched-budget methods, held-out evaluation, paired task-level statistics, and a reproducible report. The repository tests cover leakage guards, budget accounting, retrieval versioning, partial/full embedding compatibility, outcome masking, shaping arithmetic, and exact cold-start parity.

Generalization remains limited by three final datasets, short PPO budgets, a small finite search space, and a shared validation-derived score threshold. A larger independent benchmark and confirmatory experiments would be new research work. The source documents' 30% reduction is a hypothesis, not a result promised by the implementation. No private student details or original Word documents are redistributed.
