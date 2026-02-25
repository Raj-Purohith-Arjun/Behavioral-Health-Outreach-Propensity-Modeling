from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataSpec:
    rows: int
    seed: int


def build_synthetic_population(spec: DataSpec) -> pd.DataFrame:
    rng = np.random.default_rng(spec.seed)
    age = rng.integers(18, 75, spec.rows)
    prior_engagements = rng.poisson(1.7, spec.rows)
    outreach_count_90d = rng.integers(0, 8, spec.rows)
    plan_tier = rng.choice(["basic", "standard", "premium"], p=[0.45, 0.4, 0.15], size=spec.rows)
    severity_score = rng.normal(0.0, 1.0, spec.rows)
    days_since_last_contact = rng.integers(1, 120, spec.rows)
    podcast_minutes = np.clip(rng.normal(160, 80, spec.rows), 5, None)
    email_open_rate = np.clip(rng.beta(2.5, 5, spec.rows), 0, 1)

    treatment_probability = 1 / (1 + np.exp(-(-0.2 + 0.5 * email_open_rate + 0.1 * prior_engagements)))
    treatment = rng.binomial(1, treatment_probability)

    plan_effect = np.select(
        [plan_tier == "basic", plan_tier == "standard", plan_tier == "premium"],
        [-0.18, 0.02, 0.16],
    )

    base_logit = (
        -1.7
        + 0.02 * (65 - np.abs(age - 38))
        + 0.36 * np.log1p(prior_engagements)
        - 0.1 * outreach_count_90d
        - 0.009 * days_since_last_contact
        + 0.55 * email_open_rate
        + 0.23 * np.log1p(podcast_minutes)
        + 0.2 * severity_score
        + plan_effect
    )

    heterogenous_treatment_effect = (
        0.08
        + 0.08 * (plan_tier == "premium")
        + 0.06 * (email_open_rate > 0.38)
        - 0.04 * (outreach_count_90d >= 5)
    )

    treated_logit = base_logit + treatment * heterogenous_treatment_effect
    engagement_prob = 1 / (1 + np.exp(-treated_logit))
    engaged = rng.binomial(1, engagement_prob)

    return pd.DataFrame(
        {
            "member_id": np.arange(1, spec.rows + 1),
            "age": age,
            "prior_engagements": prior_engagements,
            "outreach_count_90d": outreach_count_90d,
            "plan_tier": plan_tier,
            "severity_score": severity_score,
            "days_since_last_contact": days_since_last_contact,
            "podcast_minutes": podcast_minutes,
            "email_open_rate": email_open_rate,
            "treatment": treatment,
            "engaged": engaged,
        }
    )


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(
        df[
            [
                "age",
                "prior_engagements",
                "outreach_count_90d",
                "severity_score",
                "days_since_last_contact",
                "podcast_minutes",
                "email_open_rate",
                "plan_tier",
            ]
        ],
        columns=["plan_tier"],
        dtype=float,
    )
