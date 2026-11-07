# Contributing to TRAJMETA

TRAJMETA is a research implementation of trajectory-guided AutoML. Contributions should improve correctness, reproducibility, or usability while preserving the experimental contract described in the [README](README.md).

## Development setup

Use Python 3.11 or 3.12. Create a branch for your change and a virtual environment. The recorded lockfile targets Windows with Python 3.12.

From the repository root in PowerShell:

```powershell
git switch -c fix/describe-your-change
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
```

If the lockfile cannot be resolved on your platform, install the declared development and reporting extras instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,report]"
```

Record the Python and package versions when reporting a platform-specific problem. Use the equivalent virtual-environment interpreter path on other operating systems; the recorded environment was tested on Windows.

## Run checks

Run the full test suite before submitting code changes:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

While developing a preprocessing, evaluation, or budget-accounting fix, the foundation tests provide a focused starting point:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_foundation.py -q
```

For a bug fix, first add a small regression test that fails on the existing code, then make it pass with a focused implementation change. Keep tests deterministic where possible and assert independently derived expected outcomes.

The test suite and rerunning the published study are different checks. See the README's [study reproduction instructions](README.md#verify-or-reproduce-the-transfer-study) when a change affects experimental results. Independent replications must use a new output directory.

## Preserve the experimental contract

- Fit imputation, encoding, scaling, and feature selection only within the relevant training fold.
- Do not use outer-test labels or scores to select configurations, shape rewards, or build the training bank.
- Count failed and repeated candidate evaluations toward the configured search budget.
- Preserve dataset provenance, checksums, split membership, and versioned run records.
- Do not overwrite published evidence or include private datasets, credentials, or machine-specific paths in new examples.
- Distinguish implemented behavior from research hypotheses; do not present the proposed 30% improvement as an established result.

## Report a problem or submit a pull request

A useful bug report includes the command or minimal input, expected behavior, actual output or traceback, Python/package versions, and whether synthetic or real data was used. Remove private information and keep the reproduction small.

Keep each pull request focused on one problem. Explain the before/after behavior, list the checks you ran and their outcomes, and disclose material AI assistance. Documentation-only changes do not require new automated tests, but commands, paths, and links should be checked against the repository.

For substantial changes to the search space, transfer mechanism, or study protocol, discuss the intended behavior in an issue before implementing it.
