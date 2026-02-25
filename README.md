# Behavioral Health Outreach Propensity Modeling

Production-grade machine learning and experimentation workflow for outreach prioritization. The project models member-level propensity to engage, estimates incremental treatment impact, and reports KPI trade-offs used for planning outreach capacity.

## Project goals
- Prioritize outreach to members most likely to engage.
- Quantify incremental impact with confidence intervals.
- Reduce low-value outreach while preserving business impact.
- Provide transparent metrics and reproducible outputs.

## Tech stack
- Python 3.10+
- XGBoost for propensity ranking
- SQL feature extraction template
- Bootstrap-based uplift confidence intervals
- Pytest + Ruff for quality checks

## Repository structure
- `src/outreach_model/data.py`: synthetic data and feature matrix builders.
- `src/outreach_model/model.py`: model training (XGBoost primary).
- `src/outreach_model/metrics.py`: KPI, A/B, and uplift calculations.
- `src/outreach_model/pipeline.py`: end-to-end training and report artifacts.
- `sql/outreach_feature_extract.sql`: warehouse query template.
- `configs/default.yaml`: training and experiment configuration.
- `tests/test_pipeline.py`: pipeline correctness checks.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
./scripts/run_pipeline.sh
```

Outputs are written to `artifacts/`:
- `metrics.csv`
- `scored_population.csv`

## KPI definitions
- **Engagement Lift %**: uplift in engagement rate from model-ranked outreach versus random baseline at fixed outreach capacity.
- **Low-Value Outreach Reduction %**: reduction in non-engaged treated contacts versus baseline.
- **Incremental Lift**: engagement rate difference between treated and control cohorts, with 95% CI.

## Business interpretation framework
1. Set weekly outreach capacity.
2. Rank eligible members by score.
3. Select top segment that maximizes engagement lift while staying within contact constraints.
4. Monitor incremental lift confidence intervals by cohort and plan tier.
5. Reallocate outreach resources when confidence intervals overlap with zero.

## Reproducibility
Configuration values are centralized in `configs/default.yaml`. All core metrics are generated from deterministic seeds.
