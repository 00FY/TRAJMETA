import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trajmeta.automl import Candidate, Evaluator
from trajmeta.benchmark import licence_evidence, prepare_frame
from trajmeta.cli import run
from trajmeta.data import audit_manifests, load_dataset, split_task
from trajmeta.repeat import execute


def metadata():
    return {"tasks": ["Classification"], "variables": [
        {"name": "id", "role": "ID", "type": "Integer"},
        {"name": "value", "role": "Feature", "type": "Continuous"},
        {"name": "category", "role": "Feature", "type": "Categorical"},
        {"name": "target", "role": "Target", "type": "Categorical"}]}


def test_audit_excludes_ids_and_duplicates():
    data = b'id,value,category,target\n1,1,a,yes\n2,2,b,no\n3,3,a,yes\n4,4,b,no\n5,1,a,yes\n'
    frame, report = prepare_frame(metadata(), data)
    assert "id" not in frame and report["excluded_columns"] == ["id"]
    assert report["duplicate_rows_removed"] == 1 and len(frame) == 4
    assert report["class_counts"] == {"yes": 2, "no": 2}


def test_conflicting_duplicates_rejected():
    with pytest.raises(ValueError, match="conflicting"):
        prepare_frame(metadata(), b'id,value,category,target\n1,1,a,yes\n2,1,a,no\n')


def test_licence_not_inferred_from_public_access():
    with pytest.raises(ValueError, match="explicit"):
        licence_evidence(b'public download')
    evidence = licence_evidence(b'CC BY 4.0 https://creativecommons.org/licenses/by/4.0/')
    assert evidence["licence"] == "CC BY 4.0"


def test_numeric_categories_preserved(tmp_path):
    from hashlib import sha256
    path = tmp_path / "data.csv"
    path.write_text("category,value,target\n1,2,a\n2,3,b\n")
    row = pd.Series({"source_url": "https://example.invalid", "path": str(path),
                     "sha256": sha256(path.read_bytes()).hexdigest(), "target_column": "target",
                     "categorical_columns": '["category"]'})
    X, _ = load_dataset(row, tmp_path)
    assert X.category.iloc[0] == "1"


def test_split_seed_independent_of_algorithm_seed():
    X = pd.DataFrame({"x": np.arange(40)})
    y = pd.Series([0, 1] * 20)
    one = Evaluator(X, y, 1, 2, 11, split_seed=2026)
    two = Evaluator(X, y, 1, 2, 29, split_seed=2026)
    for (a, b), (c, d) in zip(one.folds, two.folds):
        assert np.array_equal(a, c) and np.array_equal(b, d)


def test_worker_success_and_checkpoint(tmp_path):
    X = pd.DataFrame({"x": np.arange(40)})
    y = pd.Series([0, 1] * 20)
    log = tmp_path / "evaluations.jsonl"
    ev = Evaluator(X, y, 1, 2, 42, trial_timeout_seconds=30, log_path=log)
    try:
        record = ev.evaluate(Candidate("none", "all", "tree", 0))
        assert record["status"] == "ok"
        assert json.loads(log.read_text()) == record
    finally:
        ev.close()


def test_worker_timeout_consumes_budget_and_stops():
    X = pd.DataFrame({"x": np.arange(40)})
    y = pd.Series([0, 1] * 20)
    ev = Evaluator(X, y, 1, 2, 42, trial_timeout_seconds=0.000001)
    try:
        record = ev.evaluate(Candidate("none", "all", "tree", 0))
        assert record["status"] == "timeout" and len(ev.records) == 1
        assert ev.worker.process is None
    finally:
        ev.close()


def test_resume_does_not_reexecute_completed_units(tmp_path, monkeypatch):
    import yaml
    import trajmeta.repeat as module
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/pilot.yaml").read_text())
    config.update(datasets=["synthetic_0"], methods=["random"], seeds=[11, 29])
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    calls = []
    def fake_run(*args, **kwargs):
        calls.append(args)
        return [{"evaluations": 8}]
    monkeypatch.setattr(module, "run", fake_run)
    assert len(execute(path, tmp_path / "runs", root=root)) == 2
    assert len(execute(path, tmp_path / "runs", root=root)) == 2
    assert len(calls) == 2
    config["budget"] = 9
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="changed"):
        execute(path, tmp_path / "runs", root=root)


def test_frozen_manifest_tamper_rejected(tmp_path):
    import yaml
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/pilot.yaml").read_text())
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"hashes": {"dataset_manifest.csv": "invalid", "split_manifest.csv": "invalid"}}))
    config["freeze_file"] = str(freeze)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="modified"):
        run(path, tmp_path / "out", root=root)
    assert not (tmp_path / "out").exists()
