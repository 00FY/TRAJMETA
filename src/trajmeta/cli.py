"""Reproducible local pilot and trajectory-bank entry points."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import pandas as pd
import yaml

from .agents import run_optuna, run_ppo, run_random
from .automl import Candidate, Evaluator, candidates, final_score
from .data import audit_manifests, load_dataset, split_task
from .trajectories import SCHEMA_VERSION, write_bank


def json_write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config["schema_version"] != 1 or config["metric"] != "f1_macro":
        raise ValueError("Unsupported config schema or reward metric")
    if not isinstance(config["budget"], int) or config["budget"] < 1 or config["folds"] < 2:
        raise ValueError("Invalid budget or folds")
    if not 0 < config["outer_test_fraction"] < 1:
        raise ValueError("Invalid outer test fraction")
    if config["target_score"] is not None and not 0 <= config["target_score"] <= 1:
        raise ValueError("Target must be null or a preregistered score in [0, 1]")
    if not config["methods"] or not set(config["methods"]).issubset({"random", "optuna", "ppo"}):
        raise ValueError("Unknown or empty methods")
    if not config["datasets"] or len(set(config["datasets"])) != len(config["datasets"]):
        raise ValueError("Datasets must be nonempty and unique")
    candidates(config["search_space"])
    if config.get("trial_timeout_seconds") is not None and config["trial_timeout_seconds"] <= 0:
        raise ValueError("Trial timeout must be positive")
    return config


def provenance(root, config, manifest, splits):
    source_hash = sha256()
    for path in sorted((root / "src").rglob("*.py")):
        source_hash.update(path.relative_to(root).as_posix().encode())
        source_hash.update(path.read_bytes())
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"created_at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "platform": platform.platform(), "processor": platform.processor(), "git_commit": commit,
            "source_sha256": source_hash.hexdigest(), "config": config,
            "manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
            "split_manifest_sha256": sha256(splits.read_bytes()).hexdigest(),
            "packages": {d.metadata["Name"]: d.version for d in distributions()}}


def run(config_path, output, bank=False, root=None):
    root = Path(root or Path(config_path).resolve().parent.parent)
    config = load_config(config_path)
    manifest, splits = root / config["manifest"], root / config["split_manifest"]
    if config.get("freeze_file"):
        frozen = json.loads((root / config["freeze_file"]).read_text(encoding="utf-8"))
        for path in (manifest, splits):
            if sha256(path.read_bytes()).hexdigest() != frozen["hashes"][path.name]:
                raise ValueError("Frozen benchmark manifest was modified")
    inventory = audit_manifests(pd.read_csv(manifest), pd.read_csv(splits)).set_index("dataset_key")
    if not set(config["datasets"]).issubset(inventory.index):
        raise ValueError("Configured dataset missing from manifest")
    chosen = inventory.loc[config["datasets"]]
    if (chosen.split == "meta_test").any():
        raise ValueError("Meta-test is sealed in this development CLI; final evaluation is not implemented")
    if bank and not chosen.split.eq("meta_train").all():
        raise ValueError("Bank generation requires meta_train datasets exclusively")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    prov = provenance(root, config, manifest, splits)
    prov_hash = sha256(json.dumps(prov, sort_keys=True).encode()).hexdigest()
    json_write(output / "provenance.json", prov)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output / "dataset_manifest.csv").write_bytes(manifest.read_bytes())
    (output / "split_manifest.csv").write_bytes(splits.read_bytes())
    summaries, trajectories = [], []
    for key, row in chosen.iterrows():
        X, y = load_dataset(row, root)
        split_seed = config.get("split_seed", config["seed"])
        task = split_task(X, y, split_seed, config["outer_test_fraction"], config["folds"])
        methods = ["ppo"] if bank else config["methods"]
        for method in methods:
            start = perf_counter()
            directory = output / key / method
            directory.mkdir(parents=True)
            evaluator = Evaluator(task.X_train, task.y_train, config["budget"], config["folds"], config["seed"],
                                  split_seed=split_seed, trial_timeout_seconds=config.get("trial_timeout_seconds"),
                                  log_path=directory / "evaluations.jsonl")
            try:
                if method == "ppo":
                    episodes = run_ppo(evaluator, config["search_space"], config["seed"], config["ppo"], directory / "policy.zip")
                else:
                    episodes = {"random": run_random, "optuna": run_optuna}[method](evaluator, config["search_space"], config["seed"])
            finally:
                evaluator.close()
            search_seconds = perf_counter() - start
            best = evaluator.best
            summary = {"dataset_key": key, "split": row.split, "method": method, "seed": config["seed"],
                       "synthetic": row.source_url.startswith("generated:"), "evaluations": len(evaluator.records),
                       "failures": sum(r["status"] != "ok" for r in evaluator.records),
                       "best_inner_score": best["inner_score"] if best else None,
                       "search_seconds": search_seconds, "target_score": config["target_score"], "n_target": None,
                       "split_seed": split_seed}
            if config["target_score"] is not None:
                summary["n_target"] = next((r["evaluation"] for r in evaluator.records
                                            if r["status"] == "ok" and r["inner_score"] >= config["target_score"]), None)
            # The bank never needs to open outer-test labels.
            if best and not bank and config.get("report_outer_score", True):
                summary.update(final_score(Candidate(**best["pipeline_config"]), task, config["seed"]))
            for episode in episodes:
                record = {**episode, "schema_version": SCHEMA_VERSION,
                          "trajectory_id": f"{key}:{config['seed']}:{episode['evaluation']}",
                          "dataset_key": key, "split": row.split, "seed": config["seed"],
                          "policy_version": "ppo-baseline-v1", "provenance_hash": prov_hash}
                trajectories.append(record)
            json_write(directory / "evaluations.json", evaluator.records)
            json_write(directory / "summary.json", summary)
            if episodes:
                json_write(directory / "episodes.json", [r for r in trajectories if r["dataset_key"] == key])
            summaries.append(summary)
            print(f"{key:16} {method:8} evaluations={len(evaluator.records)} failures={summary['failures']} best_inner={summary['best_inner_score']}", flush=True)
    if bank:
        write_bank(output / "trajectories.jsonl", trajectories)
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    json_write(output / "summary.json", summaries)
    return summaries


def main():
    parser = argparse.ArgumentParser(description="TrajMeta-AutoML development pilot")
    parser.add_argument("command", choices=["run", "bank", "audit"])
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--output", help="New output directory (existing directories are never overwritten)")
    args = parser.parse_args()
    if args.command == "audit":
        config = load_config(args.config)
        root = Path(args.config).resolve().parent.parent
        rows = audit_manifests(pd.read_csv(root / config["manifest"]), pd.read_csv(root / config["split_manifest"]))
        print(f"Validated {len(rows)} dataset entries; provenance fields and split isolation pass.")
        print("This structural audit does not independently verify licences or semantic near-duplicates.")
        return
    output = args.output or f"results/{args.command}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    run(args.config, output, bank=args.command == "bank")


if __name__ == "__main__":
    main()
