"""Official UCI inventory and immutable, explicitly scoped pilot audit."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .data import audit_manifests, split_task

# Candidate discovery only: admission requires a separate reviewed selection.
CANDIDATES = [2, 12, 14, 15, 17, 18, 19, 22, 27, 29, 31, 33, 39, 42, 44, 45,
              46, 50, 52, 53, 59, 73, 76, 80, 81, 94, 101, 109, 110, 144,
              146, 148, 149, 151, 159, 186, 222, 236, 267, 327]
SELECTED = {
    109: ("meta_train", "wine_cultivars", "Measured wine chemistry; related wine variants must remain in this group."),
    17: ("meta_train", "wisconsin_breast_cancer", "Diagnostic nuclear measurements; remove source ID; reserve all Wisconsin variants for training."),
    52: ("meta_train", "ionosphere_radar", "Radar-return classification; one collection site; no claim of site generalization."),
    42: ("meta_val", "glass_identification", "Remove sequential ID; small minority classes; descriptive chemistry only."),
    267: ("meta_val", "banknote_authentication", "Image-derived banknote measurements; no source ID feature."),
    53: ("meta_test", "iris", "Small, easy biological benchmark; retain source version and deduplicate rows."),
    144: ("meta_test", "german_credit", "Historical credit benchmark with demographic attributes; research comparison only; no deployment or fairness claim."),
    151: ("meta_test", "sonar_mines_rocks", "Sonar frequency measurements; distinct collection from ionosphere radar; no domain-disjoint claim."),
}


def digest(value):
    return sha256(value).hexdigest()


def fetch(url):
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "archive.ics.uci.edu":
        raise ValueError("Only official HTTPS UCI downloads are allowed")
    with urlopen(url, timeout=45) as response:
        if urlparse(response.url).hostname != "archive.ics.uci.edu":
            raise ValueError("Download redirected outside UCI")
        content = response.read(20_000_001)
    if len(content) > 20_000_000:
        raise ValueError("Pilot download exceeds 20 MB")
    return content


def metadata(dataset_id):
    content = fetch(f"https://archive.ics.uci.edu/api/dataset?id={dataset_id}")
    payload = json.loads(content)
    if payload.get("status") != 200 or not payload.get("data"):
        raise ValueError(f"UCI metadata unavailable for {dataset_id}")
    return payload["data"], content


def licence_evidence(page):
    text = page.decode("utf-8")
    # Check the individual dataset page, not an assumed repository-wide licence.
    if "creativecommons.org/licenses/by/4.0" not in text or not re.search(r"CC BY 4\.0|Creative Commons Attribution 4\.0", text):
        raise ValueError("Individual page lacks explicit CC BY 4.0 evidence")
    return {"licence": "CC BY 4.0", "licence_url": "https://creativecommons.org/licenses/by/4.0/",
            "page_sha256": digest(page), "evidence": "CC BY 4.0"}


def prepare_frame(meta, raw):
    variables = meta["variables"]
    features = [v["name"] for v in variables if v["role"] == "Feature"]
    targets = [v["name"] for v in variables if v["role"] == "Target"]
    if "Classification" not in meta["tasks"] or len(targets) != 1 or not features:
        raise ValueError("Need classification with exactly one documented target")
    frame = pd.read_csv(BytesIO(raw), na_values=["?"], keep_default_na=True)
    if not set(features + targets).issubset(frame.columns):
        raise ValueError("Downloaded schema differs from metadata")
    dropped = [str(c) for c in frame if c not in features + targets]
    frame = frame[features + targets].copy()
    if len(frame) > 5000 or len(features) > 60:
        raise ValueError("Dataset exceeds the declared pilot row/feature limit")
    if frame[targets[0]].isna().any():
        raise ValueError("Missing target labels")
    categories = [v["name"] for v in variables if v["role"] == "Feature" and v["type"] in {"Categorical", "Binary"}]
    for name in categories:
        frame[name] = frame[name].astype("str").where(frame[name].notna(), np.nan)
    # Identical feature rows with conflicting labels require group-aware splitting; reject here.
    if frame.groupby(features, dropna=False)[targets[0]].nunique().gt(1).any():
        raise ValueError("Identical features have conflicting targets; grouped split required")
    duplicate_rows = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().reset_index(drop=True)
    X, y = frame[features], frame[targets[0]]
    if y.nunique() < 2 or y.value_counts().min() < 2:
        raise ValueError("Insufficient examples per class")
    hashes = sorted(pd.util.hash_pandas_object(frame, index=False).astype(str))
    return frame, {"target_column": targets[0], "categorical_columns": categories,
                   "excluded_columns": dropped, "duplicate_rows_removed": duplicate_rows,
                   "n_rows": len(frame), "n_features": len(features),
                   "class_counts": {str(k): int(v) for k, v in y.value_counts().items()},
                   "missing_fraction": float(X.isna().to_numpy().mean()),
                   "constant_columns": X.columns[X.nunique(dropna=False) <= 1].tolist(),
                   "content_fingerprint": digest("\n".join(hashes).encode())}


def inventory(output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for dataset_id in CANDIDATES:
        row = {"source_id": dataset_id, "source_url": f"https://archive.ics.uci.edu/dataset/{dataset_id}",
               "admission": "pending_review", "licence_verified": False,
               "checked_at_utc": datetime.now(timezone.utc).isoformat()}
        try:
            meta, _ = metadata(dataset_id)
            row.update(name=meta["name"], source_url=meta["repository_url"],
                       n_rows=meta["num_instances"], n_features=meta["num_features"],
                       tasks=";".join(meta["tasks"]), targets=json.dumps(meta["target_col"]),
                       domain=meta["area"], doi=meta["dataset_doi"])
            if "Classification" not in meta["tasks"]:
                row.update(admission="excluded", reason="Not a classification task")
            elif meta["num_instances"] > 5000 or (meta["num_features"] or 0) > 60:
                row.update(admission="deferred", reason="Exceeds small CPU pilot size limits")
            else:
                row["reason"] = "Needs licence, data-quality, and related-version review"
        except Exception as exc:
            row.update(admission="unavailable", reason=str(exc))
        rows.append(row)
        print(f"candidate {dataset_id}: {row['admission']}", flush=True)
    pd.DataFrame(rows).to_csv(output, index=False)


def audit(root, output):
    root, output = Path(root), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Audit output must be new or empty after an interrupted download")
    output.mkdir(parents=True, exist_ok=True)
    raw_dir, processed = root / "data/raw/uci-pilot-v1", root / "data/processed/uci-pilot-v1"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    rows, splits, reports = [], [], []
    for dataset_id, (split, group, review) in SELECTED.items():
        meta, meta_raw = metadata(dataset_id)
        page = fetch(meta["repository_url"])
        evidence = licence_evidence(page)
        if not meta.get("data_url"):
            raise ValueError(f"No canonical CSV available for {dataset_id}")
        raw = fetch(meta["data_url"])
        frame, report = prepare_frame(meta, raw)
        # Audit split feasibility, never train/evaluate a model on reserved tasks.
        split_task(frame.drop(columns=report["target_column"]), frame[report["target_column"]],
                   seed=2026, folds=2)
        key = f"uci_{dataset_id}"
        target = processed / f"{key}.csv"
        encoded = frame.to_csv(index=False, lineterminator="\n").encode()
        if target.exists() and target.read_bytes() != encoded:
            raise ValueError("Frozen processed data changed; create a new audit version")
        target.write_bytes(encoded)
        (raw_dir / f"{key}.csv").write_bytes(raw)
        (raw_dir / f"{key}.metadata.json").write_bytes(meta_raw)
        timestamp = datetime.now(timezone.utc).isoformat()
        row = {"dataset_key": key, "source_id": dataset_id, "source_url": meta["repository_url"],
               "licence": evidence["licence"], "access_date": timestamp,
               "target_column": report["target_column"], "domain": meta["area"],
               "duplicate_group": group, "task_type": "classification",
               "path": target.relative_to(root).as_posix(), "sha256": digest(encoded), "verified": True,
               "categorical_columns": json.dumps(report["categorical_columns"]),
               "raw_sha256": digest(raw), "metadata_sha256": digest(meta_raw),
               "content_fingerprint": report["content_fingerprint"], "doi": meta["dataset_doi"]}
        rows.append(row)
        splits.append({"dataset_key": key, "split": split})
        reports.append({**row, **report, **evidence, "name": meta["name"], "review": review,
                        "creators": meta["creators"], "data_url": meta["data_url"],
                        "source_last_updated": meta["last_updated"],
                        "split": split, "audit_scope": "pilot admission; no domain-disjoint claim"})
        print(f"audited {key}: {len(frame)} rows, {split}", flush=True)
    manifest, split_frame = pd.DataFrame(rows), pd.DataFrame(splits)
    audit_manifests(manifest, split_frame)
    if manifest.content_fingerprint.duplicated().any():
        raise ValueError("Exact content duplicate between pilot datasets")
    manifest.to_csv(output / "dataset_manifest.csv", index=False)
    split_frame.to_csv(output / "split_manifest.csv", index=False)
    (output / "audit.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    hashes = {name: digest((output / name).read_bytes()) for name in ["dataset_manifest.csv", "split_manifest.csv", "audit.json"]}
    (output / "freeze.json").write_text(json.dumps({"scope": "pilot-v1", "hashes": hashes,
        "split_seed": 2026, "folds": 2, "selection_basis": "size, documented role schema, reuse terms; no model scores",
        "warning": "Not a full meta-learning benchmark; related-version review remains necessary for expansion"}, indent=2), encoding="utf-8")


def restore(root, audit_path):
    """Restore ignored data without changing the committed evidence or admitting new data."""
    root, audit_path = Path(root), Path(audit_path)
    frozen = json.loads((audit_path / "freeze.json").read_text(encoding="utf-8"))
    for name, expected in frozen["hashes"].items():
        if digest((audit_path / name).read_bytes()) != expected:
            raise ValueError("Frozen audit was modified")
    reports = json.loads((audit_path / "audit.json").read_text(encoding="utf-8"))
    for report in reports:
        meta, meta_raw = metadata(report["source_id"])
        if digest(meta_raw) != report["metadata_sha256"]:
            raise ValueError("Upstream metadata changed; review a new dataset version")
        raw = fetch(report["data_url"])
        if digest(raw) != report["raw_sha256"]:
            raise ValueError("Upstream data changed; review a new dataset version")
        frame, _ = prepare_frame(meta, raw)
        encoded = frame.to_csv(index=False, lineterminator="\n").encode()
        if digest(encoded) != report["sha256"]:
            raise ValueError("Processed data differs from frozen checksum")
        path = (root / report["path"]).resolve()
        if not path.is_relative_to(root.resolve() / "data/processed"):
            raise ValueError("Unexpected restore destination")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != encoded:
            raise ValueError("Refusing to overwrite a changed local dataset")
        path.write_bytes(encoded)
        print(f"restored {report['dataset_key']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["inventory", "audit", "restore"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inventory":
        inventory(args.output)
    elif args.command == "audit":
        audit(args.root, args.output)
    else:
        restore(args.root, args.output)


if __name__ == "__main__":
    main()
