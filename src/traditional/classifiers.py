from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


def build_classifier(name: str, seed: int, params: dict[str, Any]):
    name = name.lower()
    if name in {"svm", "linearsvm", "linear_svc"}:
        return LinearSVC(
            C=float(params.get("C", 1.0)),
            class_weight=params.get("class_weight", None),
            max_iter=int(params.get("max_iter", 5000)),
            dual=bool(params.get("dual", False)),
            random_state=seed,
        )
    if name in {"rf", "random_forest", "randomforest"}:
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=params.get("max_depth", None),
            n_jobs=int(params.get("n_jobs", -1)),
            class_weight=params.get("class_weight", None),
            random_state=seed,
        )
    raise ValueError(f"Unknown classifier: {name}")


def decision_scores(model, features: np.ndarray) -> np.ndarray:
    """Return shape (N, C) scores for Top-k ranking."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        if scores.ndim == 1:
            # Binary edge case
            scores = np.column_stack([-scores, scores])
        return scores
    raise AttributeError(f"{type(model).__name__} has no ranking scores")
