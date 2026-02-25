from outreach_model.data import DataSpec, build_synthetic_population
from outreach_model.pipeline import run_training_pipeline


def test_synthetic_population_schema():
    frame = build_synthetic_population(DataSpec(rows=100, seed=7))
    assert frame.shape[0] == 100
    assert {"member_id", "engaged", "treatment", "plan_tier"}.issubset(frame.columns)


def test_pipeline_metrics_ranges(tmp_path):
    summary = run_training_pipeline("configs/default.yaml", str(tmp_path))
    assert 0.5 <= summary.roc_auc <= 1.0
    assert 0.3 <= summary.pr_auc <= 1.0
    assert summary.ci_low <= summary.incremental_lift <= summary.ci_high
