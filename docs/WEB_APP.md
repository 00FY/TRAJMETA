# Local research workspace

From the repository, run `./scripts/run_ui.ps1`, or double-click **Start TrajMeta.cmd**. Open **http://127.0.0.1:8765**. Keep the terminal open; Ctrl+C stops the server and its active search. Choose another port with `./scripts/run_ui.ps1 -Port 8766`.

The project virtual environment and published study assets must be present. Follow README setup and restore the study data with `scripts/run_study.ps1` on a fresh checkout. No Node installation, frontend build, cloud account, or additional Python packages are required.

## Connected views

- **Overview:** verified counts, learning curves with method/dataset filters, and study findings.
- **Search lab:** nine real search methods, live evaluations, cancellation, best configuration, persistent history, and per-decision rewards and retrieved source trajectories.
- **Trajectory bank:** browse and filter all 1,056 frozen training trajectories; inspect each four-decision pipeline.
- **Dataset registry:** provenance, class counts, audit metadata, and immutable split membership.
- **Study results:** compare all methods globally or by held-out dataset; download the report, run summaries, comparison CSV, and statistics JSON.

Search Lab uses the actual `trajmeta` evaluator, PPO, encoder and retrieval implementations. It writes only to ignored `results/ui/<run-id>/` folders. Saved jobs include request, evaluation log, result, and worker log. Study reports, checkpoints, protocol, and source hashes remain unchanged. Uploaded CSV content stays local in the run request; do not commit this directory. The app is loopback-only, has no authentication, and is not intended for public hosting.

## Search constraints

One worker runs at a time, with 1–32 charged evaluations and a 15-minute app limit checked during API polling. The evaluator uses two inner folds and a fixed split seed of 2026. Scores are inner-validation macro-F1, not new held-out research results. Built-in exploratory datasets are synthetic development data and meta-training tasks only; meta-validation/test tasks cannot be launched through this UI.

CSV uploads support classification with 20–5,000 rows, at most 60 features, and 2–20 classes with at least six rows per class after exact duplicate removal. The UI accepts files below 1.8 MB to leave room for JSON encoding. Select the label column, remove identifiers and leakage-prone features, and supply data suitable for ordinary stratified splitting. Conflicting labels on identical feature rows are rejected. Grouped/time-dependent datasets need a separate split design and are outside this UI's scope. Missing target values are rejected; feature preprocessing follows the core evaluator.

Cancellation terminates the worker and its evaluation children. Completed results survive reloads. Incomplete runs after a server restart are marked interrupted and are not silently resumed. Stop the original server before launching another against the same repository.

## Architecture and checks

`webapp/server.py` serves static assets and bounded JSON endpoints using Python's standard library. `webapp/worker.py` launches an isolated search process. Vanilla JavaScript renders the interface; no remote fonts, analytics, or CDN code are loaded. Host and Origin checks prevent external websites from submitting jobs; static files and downloads use explicit allowlists.

Run `./.venv/Scripts/python.exe -m pytest -q` for project and API boundary tests. The UI is kept outside `src/` to preserve the published scientific source fingerprint.

If the app was started in the background, run `./scripts/stop_ui.ps1` to stop this repository's server and worker tree. Any unfinished run is marked interrupted on the next start.

Validation: all 42 existing core tests and 15 UI/API tests pass. Browser checks covered all five views, dataset filtering, method comparisons, and a four-evaluation shaped-PPO search with source-trajectory inspection. Live API checks covered CSV search, invalid uploads, concurrent-run rejection, cancellation, and all four report exports. The frozen study verifier passed after adding the UI.
