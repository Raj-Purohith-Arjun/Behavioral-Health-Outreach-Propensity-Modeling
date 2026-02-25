from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class EvaluationSummary:
    roc_auc: float
    pr_auc: float
    engagement_lift_pct: float
    low_value_outreach_reduction_pct: float
    incremental_lift: float
    ci_low: float
    ci_high: float


def evaluate_classifier(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    return roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score)


def compute_kpis(frame: pd.DataFrame, top_n: int) -> tuple[float, float]:
    prioritized = frame.nlargest(top_n, "score")
    baseline = frame.sample(top_n, random_state=42)

    prioritized_engagement = prioritized["engaged"].mean()
    baseline_engagement = baseline["engaged"].mean()
    engagement_lift_pct = (prioritized_engagement - baseline_engagement) / baseline_engagement * 100

    low_value = lambda df: ((df["engaged"] == 0) & (df["treatment"] == 1)).mean()
    reduction = (low_value(baseline) - low_value(prioritized)) / low_value(baseline) * 100
    return engagement_lift_pct, reduction


def estimate_incremental_lift(
    frame: pd.DataFrame,
    alpha: float,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    treated = frame.loc[frame["treatment"] == 1, "engaged"]
    control = frame.loc[frame["treatment"] == 0, "engaged"]
    lift = treated.mean() - control.mean()

    rng = np.random.default_rng(seed)
    boots = np.empty(bootstrap_iterations)
    for i in range(bootstrap_iterations):
        t = rng.choice(treated, size=len(treated), replace=True).mean()
        c = rng.choice(control, size=len(control), replace=True).mean()
        boots[i] = t - c

    z = norm.ppf(1 - alpha / 2)
    std = boots.std(ddof=1)
    return lift, lift - z * std, lift + z * std
