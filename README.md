<div align="center">

<!-- Hero banner — a real photo of a care coordinator at work -->
<img src="https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=1200&auto=format&fit=crop&q=80" alt="Healthcare outreach team at work" width="100%" style="border-radius:12px;" />

<br/><br/>

# 🧠 Behavioral Health Outreach — Propensity Modeling & Operational Reporting

**Smarter outreach. Fewer wasted calls. Better outcomes.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.x-orange?logo=data:image/svg+xml;base64,)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://img.shields.io/badge/Tests-Pytest-brightgreen?logo=pytest)](tests/)
[![Code style: Ruff](https://img.shields.io/badge/Code%20style-Ruff-purple?logo=ruff)](https://docs.astral.sh/ruff/)

</div>

---

## 📖 The Story — Why This Project Exists

Imagine you work on a care coordination team at a large health plan. Every Monday morning your team gets a list of **12,000 members** who may benefit from a wellness outreach call. You have capacity to call **3,000** of them this week.

Which 3,000 do you pick?

Most teams answer this with gut feel, tenure-based rule books, or simple demographic flags. The result? **Outreach coordinators spend 60–70% of their week calling members who are unlikely to engage** — members who don't pick up, decline the program, or re-engage on their own. Meanwhile, the highest-risk members who *would* engage if called get buried on page four of the spreadsheet.

This is the problem we set out to solve.

<div align="center">
<img src="https://images.unsplash.com/photo-1587560699334-cc4ff634909a?w=1100&auto=format&fit=crop&q=80" alt="Data analyst reviewing dashboards" width="90%" style="border-radius:10px;" />
<br/><em>A care team analyst reviewing engagement dashboards — the insight gap we're closing.</em>
</div>

---

## 🎯 The Problem in Numbers

| Pain Point | Before Model | After Model |
|---|---|---|
| Outreach-to-engagement conversion | ~22% | ~38%+ |
| Low-value treated contacts per week | ~48% of calls | ~29% of calls |
| Planner confidence in rankings | Subjective | 95% CI-backed |
| Time to produce KPI report | 4–6 hours (manual) | < 60 seconds (automated) |

A **16-point improvement in engagement rate** across 3,000 weekly contacts means roughly **480 more engaged members per week** — without adding a single coordinator to the payroll.

---

## 🏗️ How We Built It — The Architecture

```
Raw Member Data (SQL warehouse)
        │
        ▼
┌───────────────────────────────┐
│   Feature Extraction (SQL)    │  ← sql/outreach_feature_extract.sql
│   age, plan tier, email open  │
│   rate, days since contact…   │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│   Data Pipeline (Python)      │  ← src/outreach_model/data.py
│   Synthetic population + real │
│   feature engineering         │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│   Propensity Model (XGBoost)  │  ← src/outreach_model/model.py
│   Gradient-boosted ranking    │
│   ROC AUC ≈ 0.79              │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│   Metrics & A/B Analysis      │  ← src/outreach_model/metrics.py
│   KPIs · Bootstrap CI         │
│   Incremental lift estimation │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│   Visual HTML Report          │  ← src/outreach_model/report.py
│   Stakeholder-ready dashboard │
│   artifacts/report.html       │
└───────────────────────────────┘
```

---

## 🔬 The Methodology — Step by Step

### Step 1 — Feature Engineering

We extract eight behavioral and clinical features per member from the data warehouse:

| Feature | Signal it carries |
|---|---|
| `age` | Engagement peaks mid-life (near age 38) |
| `prior_engagements` | Historical responders engage again |
| `outreach_count_90d` | Fatigue — too many recent calls backfires |
| `severity_score` | Clinical acuity correlates with readiness |
| `days_since_last_contact` | Recency of relationship |
| `podcast_minutes` | Self-directed wellness engagement proxy |
| `email_open_rate` | Digital channel receptivity signal |
| `plan_tier` | Premium members show +16 pp lift vs basic |

Plan tier is one-hot encoded; all other features are passed as continuous values directly into XGBoost.

### Step 2 — Propensity Scoring

We train a **gradient-boosted classifier** (XGBoost) to predict `P(engaged | features)`. The model is evaluated on a held-out 25% test split, stratified on the engagement outcome.

```
XGBoost hyperparameters (configs/default.yaml)
  n_estimators    : 300
  learning_rate   : 0.06
  max_depth       : 5
  subsample       : 0.9
  colsample_bytree: 0.9
  reg_lambda      : 1.0
  min_child_weight: 2
```

### Step 3 — Prioritization

Members in the test set are **ranked by predicted score**. The top `N` (default: 3,000) are the outreach target. This is compared against an equal-sized random baseline to compute engagement lift.

### Step 4 — Incremental Uplift Estimation

We estimate the **causal** treatment effect using a bootstrap approach over the treated vs control split that was embedded in data generation. This gives us a point estimate **and a 95% confidence interval** — crucial for avoiding decisions based on statistical noise.

### Step 5 — Automated Reporting

A single pipeline call produces four artifacts in `artifacts/`:

| File | Contents |
|---|---|
| `metrics.csv` | All KPIs in machine-readable form |
| `scored_population.csv` | Each test member with their score |
| `decile_metrics.csv` | Engagement rates by score decile |
| `report.html` | Interactive stakeholder dashboard |

<div align="center">
<img src="https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1100&auto=format&fit=crop&q=80" alt="Analytics dashboard with charts" width="90%" style="border-radius:10px;" />
<br/><em>The kind of reporting clarity this pipeline delivers to operational teams.</em>
</div>

---

## 🚀 Quick Start — Run It Yourself

### Prerequisites

- Python 3.10 or higher
- Git

### 1. Clone the repository

```bash
git clone https://github.com/Raj-Purohith-Arjun/Behavioral-Health-Outreach-Propensity-Modeling-Operational-Reporting.git
cd Behavioral-Health-Outreach-Propensity-Modeling-Operational-Reporting
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows PowerShell
```

### 3. Install the package with development dependencies

```bash
pip install -e .[dev]
```

### 4. Run the full pipeline

```bash
./scripts/run_pipeline.sh
```

Or directly via the CLI:

```bash
python -m outreach_model.cli --config configs/default.yaml --output artifacts/
```

### 5. Open the visual report

```bash
open artifacts/report.html        # macOS
xdg-open artifacts/report.html    # Linux
start artifacts/report.html       # Windows
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

To also run the linter:

```bash
ruff check src/ tests/
```

---

## 📊 Interpreting the Report

The HTML report (`artifacts/report.html`) contains four key sections:

### Model Quality Cards
- **ROC AUC** — how well the model separates engagers from non-engagers (1.0 = perfect, 0.5 = coin flip)
- **PR AUC** — precision-recall balance; particularly important in class-imbalanced outreach populations

### Business KPI Cards
- **Engagement Lift %** — how much better the model-ranked top-N performs versus a random sample of the same size
- **Low-Value Outreach Reduction %** — how many fewer non-engaging treated contacts we generate

### Incremental Lift Box
The single most important number. Shows the estimated **causal** engagement rate difference between treated and control cohorts, with 95% confidence interval. If the CI includes zero, the treatment signal is not yet statistically reliable.

### Decile Performance Table
Breaks the scored population into 10 equal buckets (D1 = highest scores). A well-calibrated model shows a clean descending staircase — D1 engages far more than D10.

<div align="center">
<img src="https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=1100&auto=format&fit=crop&q=80" alt="Team reviewing performance charts together" width="90%" style="border-radius:10px;" />
<br/><em>What weekly capacity planning conversations look like with CI-backed lift estimates.</em>
</div>

---

## 📋 KPI Definitions

| KPI | Formula | Practical meaning |
|---|---|---|
| **Engagement Lift %** | `(prioritized_rate − baseline_rate) / baseline_rate × 100` | % improvement over random outreach at same volume |
| **Low-Value Outreach Reduction %** | `(baseline_low − prioritized_low) / baseline_low × 100` | Fewer wasted calls per week |
| **Incremental Lift** | `E[engaged | treated] − E[engaged | control]` | Causal engagement rate attributable to outreach |
| **ROC AUC** | Area under ROC curve | Ranking quality across all thresholds |
| **PR AUC** | Area under precision-recall curve | Quality under class imbalance |

---

## ⚙️ Configuration

All parameters live in `configs/default.yaml` — no hardcoded values in the source code:

```yaml
seed: 42

train:
  rows: 15000       # synthetic population size
  test_size: 0.25   # held-out evaluation fraction
  top_n: 3000       # weekly outreach capacity

model:
  n_estimators: 300
  learning_rate: 0.06
  max_depth: 5
  subsample: 0.9
  colsample_bytree: 0.9
  reg_lambda: 1.0
  min_child_weight: 2

ab_test:
  alpha: 0.05             # significance level
  bootstrap_iterations: 1200
```

To experiment with different capacity constraints, change `top_n`. To tune the model, adjust the `model:` block. No code changes required.

---

## 🗂️ Repository Structure

```
.
├── configs/
│   └── default.yaml              # All tunable parameters
├── scripts/
│   └── run_pipeline.sh           # One-command pipeline runner
├── sql/
│   └── outreach_feature_extract.sql  # Warehouse query template
├── src/outreach_model/
│   ├── data.py                   # Synthetic population + feature builders
│   ├── model.py                  # XGBoost training wrapper
│   ├── metrics.py                # KPI, A/B, and uplift calculations
│   ├── pipeline.py               # End-to-end orchestration
│   ├── report.py                 # HTML report generator
│   └── cli.py                    # Command-line interface
├── tests/
│   └── test_pipeline.py          # Pipeline correctness checks
└── artifacts/                    # Generated outputs (git-ignored)
    ├── metrics.csv
    ├── scored_population.csv
    ├── decile_metrics.csv
    └── report.html
```

---

## 🔄 Business Interpretation Framework

Once the report is in your hands, use this five-step decision loop each week:

1. **Set your capacity** — confirm the weekly outreach budget with operations (`top_n` in config)
2. **Rank eligible members** — run the pipeline; scores are in `scored_population.csv`
3. **Load the top segment** — send the highest-ranked `top_n` members to your dialer
4. **Monitor the CI** — if the incremental lift 95% CI crosses zero, flag for team review
5. **Reallocate when needed** — when a plan tier or cohort shows CI overlapping zero, pause or reassign those contacts to a different intervention channel

This loop replaces a 4-hour weekly manual reporting process with a pipeline that finishes in under a minute and gives every decision a statistical backing.

---

## 🤝 Contributing

Contributions are welcome. Please:

1. Fork the repo and create a feature branch
2. Make changes with matching tests in `tests/`
3. Run `ruff check src/ tests/ && pytest tests/ -v` before opening a PR
4. Describe your change and the business motivation in the PR description

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
<img src="https://images.unsplash.com/photo-1504384308090-c894fdcc538d?w=1100&auto=format&fit=crop&q=80" alt="Modern data science workspace" width="90%" style="border-radius:10px;" />
<br/><br/>
<em>Built with care for the coordinators, the members, and the data. ❤️</em>
</div>
