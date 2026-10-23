import json
from pathlib import Path

import pytest
import yaml

from trajmeta.bank_from_runs import build


def test_bank_export_rejects_reserved_tasks(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/pilot.yaml").read_text())
    config["datasets"] = ["synthetic_test"]
    (tmp_path / "plan.json").write_text(json.dumps({"config": config}))
    with pytest.raises(ValueError, match="held-out"):
        build(root, tmp_path, tmp_path / "bank.jsonl")
    assert not (tmp_path / "bank.jsonl").exists()


def test_bank_export_requires_completed_runs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/pilot.yaml").read_text())
    config["datasets"] = ["synthetic_0"]
    (tmp_path / "plan.json").write_text(json.dumps({"config": config}))
    with pytest.raises(ValueError, match="incomplete"):
        build(root, tmp_path, tmp_path / "bank.jsonl")
