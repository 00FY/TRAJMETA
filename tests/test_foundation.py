import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from trajmeta.agents import rollout_sizes, run_optuna, run_ppo, run_random
from trajmeta.automl import BudgetExhausted, Candidate, Evaluator, build_pipeline, candidates
from trajmeta.cli import load_config, run
from trajmeta.data import audit_manifests, load_dataset, split_task
from trajmeta.env import PipelineEnv
from trajmeta.shaping import gated_potential, shape_reward
from trajmeta.trajectories import read_bank, validate_trajectory, write_bank

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return load_config(ROOT / "configs/pilot.yaml")


@pytest.fixture
def task():
    row = pd.read_csv(ROOT / "data/dataset_manifest.csv").iloc[0]
    return split_task(*load_dataset(row, ROOT))


def evaluator(task, budget=2):
    return Evaluator(task.X_train, task.y_train, budget=budget, folds=2, seed=42)


def test_split_integrity():
    manifest = pd.read_csv(ROOT / "data/dataset_manifest.csv")
    splits = pd.read_csv(ROOT / "data/split_manifest.csv")
    assert len(audit_manifests(manifest, splits)) == 5
    manifest.loc[4, "duplicate_group"] = manifest.loc[0, "duplicate_group"]
    with pytest.raises(ValueError, match="Duplicate groups"):
        audit_manifests(manifest, splits)


def test_duplicate_ids_rejected():
    manifest = pd.read_csv(ROOT / "data/dataset_manifest.csv")
    splits = pd.read_csv(ROOT / "data/split_manifest.csv")
    with pytest.raises(ValueError, match="unique"):
        audit_manifests(manifest, pd.concat([splits, splits.iloc[:1]]))


def test_pipeline_fit_only_training_fold(task):
    pipe = build_pipeline(Candidate("standard", "all", "logistic", 0), 42)
    ev = evaluator(task)
    a, b = ev.folds[0]
    X = task.X_train.copy()
    X.iloc[b, 0] = 1e9
    fitted = clone(pipe).fit(X.iloc[a], task.y_train.iloc[a])
    imputer = fitted.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
    assert imputer.statistics_[0] == pytest.approx(X.iloc[a, 0].median())
    scaler = fitted.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scale"]
    assert scaler.mean_[0] == pytest.approx(X.iloc[a, 0].mean())
    assert not hasattr(pipe.named_steps["preprocess"], "transformers_")


def test_unknown_categories_and_capped_selection():
    X = pd.DataFrame({"x": [1., 2., 3., np.nan, 5., 6.], "cat": ["a", "b", "a", "b", "a", "b"]})
    model = build_pipeline(Candidate("standard", "k10", "logistic", 1), 42).fit(X, [0, 0, 1, 0, 1, 1])
    assert len(model.predict(pd.DataFrame({"x": [999.], "cat": ["unseen"]}))) == 1


def test_budget_enforced(task):
    ev = evaluator(task, 1)
    candidate = Candidate("none", "all", "tree", 0)
    ev.evaluate(candidate)
    with pytest.raises(BudgetExhausted):
        ev.evaluate(candidate)
    assert len(ev.records) == 1


def test_failure_consumes_budget(task):
    ev = evaluator(task, 1)
    ev.X = ev.X.astype(str)
    # Deliberate invalid category object causes preprocessing failure.
    ev.X.iloc[0, 0] = None
    from unittest.mock import patch
    with patch("trajmeta.automl.cross_val_score", side_effect=ValueError("deliberate failure")):
        assert ev.evaluate(Candidate("none", "all", "tree", 0))["status"] == "failed"
    assert len(ev.records) == 1 and ev.best is None


def test_environment_and_trajectory(task, config, tmp_path):
    env = PipelineEnv(evaluator(task, 1), config["search_space"])
    obs, _ = env.reset(seed=42)
    assert env.observation_space.contains(obs)
    for i in range(4):
        obs, reward, done, truncated, _ = env.step(0)
        assert env.observation_space.contains(obs)
        assert done == (i == 3) and not truncated
        if i < 3:
            assert reward == 0 and not env.evaluator.records
    with pytest.raises(RuntimeError):
        env.step(0)
    record = {**env.episodes[0], "schema_version": 1, "trajectory_id": "test:1", "dataset_key": "x",
              "split": "meta_train", "seed": 42, "policy_version": "v1", "provenance_hash": "test"}
    write_bank(tmp_path / "bank.jsonl", [record])
    assert read_bank(tmp_path / "bank.jsonl") == [record]
    record["split"] = "meta_test"
    with pytest.raises(ValueError, match="meta-training"):
        validate_trajectory(record)


def test_gymnasium_contract(task, config):
    from gymnasium.utils.env_checker import check_env
    check_env(PipelineEnv(evaluator(task, 10), config["search_space"]), skip_render_check=True)


def test_outer_labels_cannot_change_search(task, config):
    first = evaluator(task)
    run_random(first, config["search_space"], 42)
    task.y_test[:] = 99
    task.X_test.iloc[:, 0] = 1e12
    second = evaluator(task)
    run_random(second, config["search_space"], 42)
    assert [r["inner_score"] for r in first.records] == [r["inner_score"] for r in second.records]


@pytest.mark.parametrize("method", [run_random, run_optuna])
def test_baseline_budget_and_reproducibility(task, config, method):
    first, second = evaluator(task), evaluator(task)
    method(first, config["search_space"], 42)
    method(second, config["search_space"], 42)
    assert len(first.records) == len(second.records) == 2
    assert [r["pipeline_config"] for r in first.records] == [r["pipeline_config"] for r in second.records]
    assert [r["inner_score"] for r in first.records] == [r["inner_score"] for r in second.records]


@pytest.mark.parametrize("budget", [1, 3, 8, 17])
def test_ppo_rollout_exactness(config, budget):
    total, steps, batch = rollout_sizes(budget, config["ppo"])
    assert total == budget * 4 and total % steps == 0 and steps % batch == 0


def test_ppo_actual_run(task, config):
    ev = evaluator(task, 3)
    episodes = run_ppo(ev, config["search_space"], 42, config["ppo"])
    assert len(episodes) == len(ev.records) == 3
    assert all(r["status"] == "ok" for r in ev.records)


def test_search_space(config):
    assert len(candidates(config["search_space"])) == 48


def test_cold_start_and_zero_strength():
    assert gated_potential([], .8) == 0
    assert gated_potential([.1, .2], .8) == 0
    assert shape_reward(.7, 0, 0, strength=1)["shaped_reward"] == .7
    assert shape_reward(.7, .9, .2, strength=0)["shaped_reward"] == .7
    assert 0 <= gated_potential([.8, .9, 1], .8) <= 1


def test_terminal_shaping_telescopes():
    gamma, strength = .9, .4
    potentials = [.3, .8, .2, .7]
    deltas = [shape_reward(0, potentials[i], potentials[i+1], gamma=gamma, strength=strength,
                           terminal=i == 2)["shaping_difference"] for i in range(3)]
    assert sum(gamma ** i * d for i, d in enumerate(deltas)) == pytest.approx(-strength * potentials[0])


@pytest.mark.parametrize("dataset,bank", [("synthetic_test", False), ("synthetic_val", True)])
def test_cli_seals_splits(config, tmp_path, dataset, bank):
    import yaml
    config["datasets"] = [dataset]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="sealed|meta_train"):
        run(path, tmp_path / "out", bank=bank, root=ROOT)
    assert not (tmp_path / "out").exists()
