"""
StandardScaler + SVC pipeline fit on frozen ConvNeXt embeddings.

Anti-leakage contract: fit() must only ever be called with train_dev
embeddings (train + inner_val). holdout embeddings are for .predict() only,
never .fit()/.fit_transform().
"""

from typing import Any, List

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_svm_pipeline(
    kernel: str = "rbf", C: float = 1.0, gamma: Any = "scale"
) -> Pipeline:
    """StandardScaler -> SVC(probability=True), bundled so scaler stats never leak across fit calls."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel=kernel, C=C, gamma=gamma, probability=True)),
        ]
    )


def fit_svm_pipeline(
    train_dev_embeddings: np.ndarray,
    train_dev_labels: np.ndarray,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: Any = "scale",
) -> Pipeline:
    """Fit StandardScaler+SVC on train_dev embeddings ONLY. Never pass holdout data here."""
    pipeline = build_svm_pipeline(kernel=kernel, C=C, gamma=gamma)
    pipeline.fit(train_dev_embeddings, train_dev_labels)
    return pipeline


def tune_svm_hyperparams(
    train_dev_embeddings: np.ndarray,
    train_dev_labels: np.ndarray,
    C_grid: List[float],
    gamma_grid: List[Any],
    kernel: str = "rbf",
    n_folds: int = 5,
    seed: int = 42,
) -> Pipeline:
    """
    StratifiedKFold GridSearchCV run ENTIRELY INSIDE train_dev (never touches
    holdout) to pick C/gamma. Returns the refit best-estimator pipeline.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    pipeline = build_svm_pipeline(kernel=kernel)
    param_grid = {"svc__C": C_grid, "svc__gamma": gamma_grid}

    search = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1)
    search.fit(train_dev_embeddings, train_dev_labels)

    print(f"Best SVM params (train_dev CV, f1_macro): {search.best_params_}")
    print(f"Best CV f1_macro: {search.best_score_:.4f}")

    return search.best_estimator_
