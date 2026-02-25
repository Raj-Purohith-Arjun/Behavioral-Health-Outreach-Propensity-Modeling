from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


@dataclass(frozen=True)
class ModelConfig:
    n_estimators: int
    learning_rate: float
    max_depth: int
    subsample: float
    colsample_bytree: float
    reg_lambda: float
    min_child_weight: float
    random_state: int


def fit_propensity_model(x_train: np.ndarray, y_train: np.ndarray, cfg: ModelConfig):
    if XGBClassifier is not None:
        model = XGBClassifier(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_lambda=cfg.reg_lambda,
            min_child_weight=cfg.min_child_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=cfg.random_state,
        )
    else:
        model = LogisticRegression(max_iter=1500, random_state=cfg.random_state)

    model.fit(x_train, y_train)
    return model
