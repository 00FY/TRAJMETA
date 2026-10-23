"""All fitted preprocessing lives inside the cross-validation pipeline."""
from dataclasses import asdict, dataclass
from itertools import product
from time import perf_counter

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class Candidate:
    scaler: str
    selection: str
    classifier: str
    level: int


def candidates(space):
    menus = [space[k] for k in ["scalers", "selection", "classifiers", "hyperparameter_levels"]]
    if not all(menus) or any(len(set(m)) != len(m) for m in menus):
        raise ValueError("Search menus must be nonempty without duplicate entries")
    valid = [{"none", "standard", "minmax"}, {"all", "k10"},
             {"logistic", "tree", "forest", "svc"}, {0, 1}]
    if any(not set(m).issubset(v) for m, v in zip(menus, valid)):
        raise ValueError("Unsupported search-space option")
    return [Candidate(*values) for values in product(*menus)]


class CappedSelectKBest(TransformerMixin, BaseEstimator):
    def __init__(self, k=10):
        self.k = k

    def fit(self, X, y):
        self.selector_ = SelectKBest(f_classif, k=min(self.k, X.shape[1])).fit(X, y)
        return self

    def transform(self, X):
        return self.selector_.transform(X)


def build_pipeline(candidate, seed):
    scales = {"none": "passthrough", "standard": StandardScaler(), "minmax": MinMaxScaler()}
    models = {
        "logistic": LogisticRegression(C=[0.1, 1.0][candidate.level], max_iter=1000, random_state=seed),
        "tree": DecisionTreeClassifier(max_depth=[3, 8][candidate.level], random_state=seed),
        "forest": RandomForestClassifier(n_estimators=40, max_depth=[4, None][candidate.level],
                                         random_state=seed, n_jobs=1),
        "svc": SVC(C=[0.5, 2.0][candidate.level], kernel="rbf"),
    }
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", scales[candidate.scaler])])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocess = ColumnTransformer([
        ("numeric", numeric, make_column_selector(dtype_include=np.number)),
        ("categorical", categorical, make_column_selector(dtype_exclude=np.number)),
    ])
    return Pipeline([("preprocess", preprocess),
                     ("select", CappedSelectKBest() if candidate.selection == "k10" else "passthrough"),
                     ("model", models[candidate.classifier])])


class BudgetExhausted(RuntimeError):
    pass


class Evaluator:
    """This object receives no outer-test data. Failed/duplicate trials also consume budget."""
    def __init__(self, X_train, y_train, budget, folds, seed, split_seed=None, trial_timeout_seconds=None, log_path=None):
        if budget < 1 or folds < 2:
            raise ValueError("Budget must be positive and folds at least two")
        self.X, self.y, self.budget, self.seed = X_train, y_train, budget, seed
        self.folds = list(StratifiedKFold(folds, shuffle=True, random_state=seed if split_seed is None else split_seed).split(X_train, y_train))
        self.records = []
        self.log_path = log_path
        self.worker = None
        if trial_timeout_seconds is not None:
            from .timed import CVWorker
            self.worker = CVWorker(self.X, self.y, self.folds, seed, trial_timeout_seconds)

    def close(self):
        if self.worker is not None:
            self.worker.close()

    @property
    def best(self):
        valid = [r for r in self.records if r["status"] == "ok"]
        return max(valid, key=lambda r: r["inner_score"], default=None)

    def evaluate(self, candidate):
        if len(self.records) >= self.budget:
            raise BudgetExhausted("No evaluations remaining")
        start = perf_counter()
        record = {"evaluation": len(self.records) + 1, "pipeline_config": asdict(candidate)}
        try:
            if self.worker is None:
                scores = cross_val_score(build_pipeline(candidate, self.seed), self.X, self.y,
                                         cv=self.folds, scoring="f1_macro", error_score="raise", n_jobs=1)
            else:
                result = self.worker.evaluate(candidate)
                record["warnings"] = result["warnings"]
                if result["error"] is not None:
                    record.update(status=result["status"], inner_score=0.0, fold_scores=[], error=result["error"])
                    return self._finish(record, start)
                scores = np.asarray(result["scores"])
            if not np.isfinite(scores).all():
                raise ValueError("Non-finite validation score")
            record.update(status="ok", inner_score=float(scores.mean()), fold_scores=scores.tolist(), error=None)
        except (ValueError, TypeError, FloatingPointError) as exc:
            record.update(status="failed", inner_score=0.0, fold_scores=[], error=str(exc))
        return self._finish(record, start)

    def _finish(self, record, start):
        record["elapsed_seconds"] = perf_counter() - start
        self.records.append(record)
        if self.log_path is not None:
            import json
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
        return record


def final_score(candidate, task, seed):
    """Called once per method after search; never supplies rewards or selects configurations."""
    model = build_pipeline(candidate, seed).fit(task.X_train, task.y_train)
    prediction = model.predict(task.X_test)
    return {"outer_macro_f1": float(f1_score(task.y_test, prediction, average="macro")),
            "outer_balanced_accuracy": float(balanced_accuracy_score(task.y_test, prediction))}
