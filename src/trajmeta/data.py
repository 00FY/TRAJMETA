"""Dataset provenance, disjoint splits, and training-only profiles."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def audit_manifests(manifest, splits):
    required = {"dataset_key", "source_id", "source_url", "licence", "access_date",
                "target_column", "domain", "duplicate_group", "task_type", "path", "sha256", "verified"}
    if not required.issubset(manifest.columns) or not {"dataset_key", "split"}.issubset(splits.columns):
        raise ValueError("Manifest columns are missing")
    for frame in (manifest, splits):
        if frame.dataset_key.isna().any() or frame.dataset_key.duplicated().any():
            raise ValueError("Dataset IDs must be unique and nonempty")
    if set(manifest.dataset_key) != set(splits.dataset_key):
        raise ValueError("Every dataset must have exactly one frozen split")
    if not set(splits.split).issubset({"meta_train", "meta_val", "meta_test"}):
        raise ValueError("Unknown split")
    merged = manifest.merge(splits, on="dataset_key", validate="one_to_one")
    for column in required - {"path", "sha256"}:
        if merged[column].isna().any() or merged[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing provenance: {column}")
    if not merged.verified.astype(str).str.lower().eq("true").all():
        raise ValueError("Unverified datasets cannot be used")
    if not merged.task_type.eq("classification").all():
        raise ValueError("Only classification is supported")
    if (merged.groupby("duplicate_group").split.nunique() > 1).any():
        raise ValueError("Duplicate groups cross dataset splits")
    # Identical downloaded files and source identities must not cross splits either.
    for column in ["sha256", "source_url"] + (["content_fingerprint"] if "content_fingerprint" in merged else []):
        groups = merged.dropna(subset=[column])
        if column == "source_url":
            groups = groups[~groups.source_url.str.startswith("generated:")]
        if (groups.groupby(column).split.nunique() > 1).any():
            raise ValueError(f"Repeated {column} crosses splits")
    return merged


def load_dataset(row, root):
    if row.source_url == "generated:sklearn.make_classification":
        X, y = make_classification(n_samples=160, n_features=8, n_informative=5,
                                  n_redundant=1, random_state=100 + int(row.source_id))
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(8)])
        X["category"] = np.where(X.x0 > 0, "high", "low")
        X.loc[::17, "x1"] = np.nan
        return X, pd.Series(y, name="target")
    path = (Path(root) / str(row.path)).resolve()
    if not path.is_file() or sha256(path.read_bytes()).hexdigest() != str(row.sha256):
        raise ValueError("Dataset file missing or SHA256 differs from the manifest")
    frame = pd.read_csv(path)
    if "categorical_columns" in row.index and pd.notna(row.categorical_columns):
        import json
        for name in json.loads(row.categorical_columns):
            if name not in frame:
                raise ValueError("Categorical schema column missing")
            frame[name] = frame[name].astype("str").where(frame[name].notna(), np.nan)
    if row.target_column not in frame:
        raise ValueError("Target column missing")
    return frame.drop(columns=[row.target_column]), frame[row.target_column]


@dataclass
class TaskSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def split_task(X, y, seed=42, test_size=0.25, folds=3):
    if len(X) != len(y) or X.shape[1] == 0 or y.isna().any():
        raise ValueError("Invalid features or missing targets")
    if y.nunique() < 2 or y.value_counts().min() < 2:
        raise ValueError("At least two classes and two examples per class are required")
    a, b, c, d = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    if c.value_counts().min() < folds:
        raise ValueError("Not enough training examples per class for inner folds")
    return TaskSplit(a.reset_index(drop=True), b.reset_index(drop=True),
                     c.reset_index(drop=True), d.reset_index(drop=True))


def profile(X, y):
    """Fixed transforms, no fitted normalization; only pass the outer training split."""
    numeric = len(X.select_dtypes(include="number").columns) / X.shape[1]
    return np.asarray([np.log1p(len(X)) / 20, np.log1p(X.shape[1]) / 20,
                       numeric, 1 - numeric, X.isna().to_numpy().mean(),
                       np.log1p(y.nunique()) / 20, y.value_counts(normalize=True).max()],
                      dtype=np.float32).clip(0, 1)
