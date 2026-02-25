from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from .data import DataSpec, build_feature_matrix, build_synthetic_population
from .metrics import EvaluationSummary, compute_kpis, estimate_incremental_lift, evaluate_classifier
from .model import ModelConfig, fit_propensity_model


def run_training_pipeline(config_path: str, output_dir: str) -> EvaluationSummary:
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    frame = build_synthetic_population(DataSpec(rows=cfg["train"]["rows"], seed=cfg["seed"]))
    x = build_feature_matrix(frame)
    y = frame["engaged"]

    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        x,
        y,
        frame.index,
        test_size=cfg["train"]["test_size"],
        random_state=cfg["seed"],
        stratify=y,
    )

    model_cfg = ModelConfig(**cfg["model"])
    model = fit_propensity_model(x_train.values, y_train.values, model_cfg)
    scores = model.predict_proba(x_test.values)[:, 1]

    roc_auc, pr_auc = evaluate_classifier(y_test.values, scores)

    eval_frame = frame.loc[idx_test, ["member_id", "engaged", "treatment"]].copy()
    eval_frame["score"] = scores

    lift_pct, low_value_reduction = compute_kpis(eval_frame, top_n=cfg["train"]["top_n"])
    incremental_lift, ci_low, ci_high = estimate_incremental_lift(
        eval_frame,
        alpha=cfg["ab_test"]["alpha"],
        bootstrap_iterations=cfg["ab_test"]["bootstrap_iterations"],
        seed=cfg["seed"],
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    eval_frame.to_csv(output / "scored_population.csv", index=False)

    summary = EvaluationSummary(
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        engagement_lift_pct=lift_pct,
        low_value_outreach_reduction_pct=low_value_reduction,
        incremental_lift=incremental_lift,
        ci_low=ci_low,
        ci_high=ci_high,
    )

    summary_table = pd.DataFrame(
        {
            "metric": [
                "roc_auc",
                "pr_auc",
                "engagement_lift_pct",
                "low_value_outreach_reduction_pct",
                "incremental_lift",
                "incremental_lift_ci_low",
                "incremental_lift_ci_high",
            ],
            "value": [
                summary.roc_auc,
                summary.pr_auc,
                summary.engagement_lift_pct,
                summary.low_value_outreach_reduction_pct,
                summary.incremental_lift,
                summary.ci_low,
                summary.ci_high,
            ],
        }
    )
    summary_table.to_csv(output / "metrics.csv", index=False)
    return summary
