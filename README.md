# TrajMeta-AutoML

A working first milestone for the supplied TrajMeta-AutoML research design: search for tabular classification pipelines, compare fixed-budget baselines, and record PPO construction trajectories for later transfer learning.

**Status: foundation and initial real-data audit implemented; transfer hypothesis not yet evaluated.** Synthetic fixtures remain for regression tests. The real-data milestone adds an eight-dataset UCI pilot admission audit, frozen splits, repeated-seed baselines, trial deadlines, and a training-only bank exporter. See [the real-data protocol](docs/REAL_DATA_PROTOCOL.md) and [dataset attribution](docs/DATA_ATTRIBUTION.md). Neither pilot establishes transfer benefit or the proposed 30% reduction in evaluations.

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

The lockfile records the tested Windows/Python 3.12 environment. If a different platform cannot resolve it, install `-e ".[dev]"`, run the tests, and freeze a separate lockfile. No execution-policy change is required. The project is stored at `D:\Desktop\TRAJMETA` and backed up at https://github.com/00FY/TRAJMETA. After setup, `scripts/run_local.ps1` runs the pilot using the project-local `.venv`. Generated results and model checkpoints stay local and are excluded from Git; regenerate them with the commands above.

Output directories must be new. Runs never overwrite prior evidence. Default directories have unique UTC timestamps. `configs/pilot.yaml` selects three generated training tasks, eight pipeline evaluations per method, three inner folds, and one seed. PPO uses small networks and short updates for a smoke test, not a tuned research baseline.

## What works

- Manifest audit: unique dataset IDs, frozen split membership, duplicate groups, checksums/source duplication across splits, and provenance fields.
- Generated mixed numeric/categorical data with missing numeric values; audited local CSVs with SHA256 verification.
- Stratified outer train/test split; fixed inner folds shared across methods.
- Forty-eight pipelines combining three scalers, two selection modes, four classifiers, and two hyperparameter levels.
- Random Search, Optuna TPE over the same finite configuration menu, and Stable-Baselines3 PPO.
- Four-stage Gymnasium episodes with one evaluation per episode; terminal macro-F1 reward and zero intermediate reward.
- Versioned complete trajectory records, model checkpoints, per-trial outcomes, timing, package versions, source hash, and config/manifest snapshots.
- Training-only bank admission, explicit meta-test blocking, and tested shaping arithmetic for the next milestone.

## Experimental contract

The evaluator only receives outer-training rows. Imputation, categorical encoding, scaling, and selection are fitted separately in every inner training fold. When reporting is enabled, outer-test scores are computed once for the best successful candidate after each baseline search. Real-data development configurations disable this reporting. Bank generation never evaluates outer-test scores. No outer score is an observation or reward.

All methods consume exactly the configured number of evaluations, including repeat configurations and failed candidates. No method gets free cached scores. Final fitting is additional work outside the search budget. The summary's `search_seconds` includes controller overhead and PPO checkpoint writing; per-evaluation times are also logged. Runs are sequential CPU experiments. Real-data configurations isolate CV in a spawned worker with a per-trial wall-time limit; failed and timed-out trials count toward the budget. There is no hard RAM limit, so use the audited dataset size bounds.

PPO receives a flat discrete action space whose size is the least common multiple of stage menu lengths. At each stage, action modulo menu length determines the choice. Every valid choice has the same number of aliases. The default has twelve action IDs and 48 complete configurations. The observation combines seven bounded training-data descriptors, one-hot partial choices, a stage indicator, the previous action, used-budget fraction, and best inner score. Dataset IDs are excluded. Dataset descriptors use fixed transforms, not statistics fitted on held-out datasets. The exact order is in `env.py`.

PPO rollout sizes divide the total transition budget exactly. Reset starts a new construction episode but does not replenish the evaluator's run budget. After the budget is exhausted, further evaluations raise an error. This guards against the full-rollout rounding behavior documented by [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html). The environment uses the [Gymnasium reset/step API](https://gymnasium.farama.org/introduction/create_custom_env/).

`target_score` is null by default, so no N90 or N_target claim is made. Set a fixed validation-derived target only after preregistration. N_target counts successful inner validation scores reaching that threshold; unreached targets remain null and must be treated as censored, not zero.

## Real data onboarding

Add one row per audited dataset to `data/dataset_manifest.csv` and exactly one matching split row to `data/split_manifest.csv`. Local CSV paths are relative to the project root. Record the original source URL/ID, verified licence, access date, target, domain, duplicate group, and SHA256 of the exact CSV. Set `verified=true` only after independently checking the individual source's reuse terms. The structural audit cannot establish legal permission or discover semantic near-duplicates for you. Shared source URLs are conservatively grouped; use dataset-specific canonical URLs.

The loader checks the checksum before loading. Missing target labels and insufficient class counts are rejected. CSV feature dtypes are inferred by pandas: inspect IDs, numeric-coded categories, date fields, and target leakage manually before admitting a real dataset. The generated fixtures are explicitly marked `generated`; their recipe is fixed in source and tracked by the source hash. They are not UCI/OpenML benchmark entries.

## Next milestones

1. Expand the initial eight-dataset audit toward the full research pool, including OpenML mirror and related-version checks. The frozen pilot split is not the final generalization benchmark.
2. Generate a larger training-only PPO bank with multiple seeds. Assess PPO learning against Random Search and Optuna before proceeding.
3. Train the shared LSTM using within-dataset quality ranks and held-out training trajectories. Check partial/full embedding compatibility and collapse, including a simple non-learned retrieval baseline.
4. Add metadata shortlisting, trajectory retrieval, and validation-selected gates. Freeze the encoder/index and potential definition for each run. The shaping math is present, but no learned retriever or shaped PPO is claimed yet.
5. Implement transfer ablations, repeated-run aggregation, confidence intervals, and a separately guarded final evaluation command. Only then unseal meta-test.
6. Produce the research report and optional dashboard after evidence exists.

The supplied exact implementation guide is used as the more specific implementation specification. The earlier proposal's dashboard and 30% goal remain future work/hypotheses. No benchmark datasets, private student details, or original Word files are redistributed in this project.
