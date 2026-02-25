from __future__ import annotations

from pathlib import Path

import pandas as pd

from .metrics import EvaluationSummary


def _format_pct(value: float) -> str:
    return f"{value:.2f}%"


def _format_point(value: float) -> str:
    return f"{value:.4f}"


def _build_decile_table(eval_frame: pd.DataFrame) -> pd.DataFrame:
    ranked = eval_frame.sort_values("score", ascending=False).reset_index(drop=True)
    ranked["decile"] = pd.qcut(ranked.index + 1, q=10, labels=[str(i) for i in range(1, 11)])

    summary = (
        ranked.groupby("decile", observed=True)
        .agg(
            members=("member_id", "count"),
            avg_score=("score", "mean"),
            engagement_rate=("engaged", "mean"),
            treated_share=("treatment", "mean"),
        )
        .reset_index()
    )
    summary["engagement_rate_pct"] = summary["engagement_rate"] * 100
    return summary


def _render_metric_card(label: str, value: str, subtitle: str) -> str:
    return (
        '<div class="card">'
        f"<h3>{label}</h3>"
        f"<p class='value'>{value}</p>"
        f"<p class='sub'>{subtitle}</p>"
        "</div>"
    )


def _render_decile_bars(table: pd.DataFrame) -> str:
    rows = []
    for _, row in table.iterrows():
        width = max(2, min(100, row["engagement_rate_pct"]))
        rows.append(
            "<tr>"
            f"<td>D{row['decile']}</td>"
            f"<td>{row['members']}</td>"
            f"<td>{row['avg_score']:.3f}</td>"
            f"<td>{row['engagement_rate_pct']:.2f}%</td>"
            f"<td><div class='bar-wrap'><div class='bar' style='width:{width}%;'></div></div></td>"
            "</tr>"
        )
    return "\n".join(rows)


def write_visual_report(
    output_dir: str,
    summary: EvaluationSummary,
    eval_frame: pd.DataFrame,
    top_n: int,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    deciles = _build_decile_table(eval_frame)
    prioritized = eval_frame.nlargest(top_n, "score")
    baseline = eval_frame.sample(top_n, random_state=42)

    prioritized_rate = prioritized["engaged"].mean() * 100
    baseline_rate = baseline["engaged"].mean() * 100

    cards = "\n".join(
        [
            _render_metric_card("Model ROC AUC", _format_point(summary.roc_auc), "ranking quality"),
            _render_metric_card("Model PR AUC", _format_point(summary.pr_auc), "precision-recall balance"),
            _render_metric_card(
                "Engagement Lift",
                _format_pct(summary.engagement_lift_pct),
                f"top {top_n:,} vs random baseline",
            ),
            _render_metric_card(
                "Low-Value Outreach Reduction",
                _format_pct(summary.low_value_outreach_reduction_pct),
                "fewer non-engaging treated contacts",
            ),
        ]
    )

    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Outreach Propensity Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .subtitle {{ color: #4b5563; margin-top: 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(220px, 1fr)); gap: 12px; margin: 16px 0 20px; }}
    .card {{ border: 1px solid #d1d5db; border-radius: 10px; padding: 12px; background: #f9fafb; }}
    .card h3 {{ margin: 0 0 8px; font-size: 14px; color: #374151; }}
    .value {{ font-size: 24px; font-weight: 700; margin: 0; }}
    .sub {{ margin: 6px 0 0; color: #6b7280; font-size: 12px; }}
    .callout {{ border-left: 4px solid #10b981; padding: 8px 12px; background: #ecfdf5; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; text-align: left; padding: 8px; font-size: 13px; }}
    th {{ color: #111827; background: #f3f4f6; }}
    .bar-wrap {{ height: 12px; background: #e5e7eb; border-radius: 99px; overflow: hidden; }}
    .bar {{ height: 12px; background: #1db954; border-radius: 99px; }}
    .small {{ font-size: 12px; color: #6b7280; }}
  </style>
</head>
<body>
  <h1>Behavioral Health Outreach Propensity Modeling</h1>
  <p class="subtitle">Interview-ready summary of model quality, experimentation impact, and prioritization outcomes.</p>

  <div class="callout">
    <strong>Incremental Lift:</strong> {_format_point(summary.incremental_lift)}
    (95% CI: {_format_point(summary.ci_low)} to {_format_point(summary.ci_high)})
  </div>

  <div class="grid">
    {cards}
  </div>

  <h2>Outreach Prioritization Comparison</h2>
  <table>
    <tr><th>Segment</th><th>Members</th><th>Engagement Rate</th></tr>
    <tr><td>Prioritized (Top Scores)</td><td>{top_n:,}</td><td>{prioritized_rate:.2f}%</td></tr>
    <tr><td>Random Baseline</td><td>{top_n:,}</td><td>{baseline_rate:.2f}%</td></tr>
  </table>

  <h2>Score Decile Performance</h2>
  <table>
    <tr><th>Decile</th><th>Members</th><th>Avg Score</th><th>Engagement Rate</th><th>Visual</th></tr>
    {_render_decile_bars(deciles)}
  </table>

  <p class="small">Decile 1 = highest model score group. Report generated from the scored test population.</p>
</body>
</html>
"""

    report_path = output_path / "report.html"
    report_path.write_text(html.strip() + "\n", encoding="utf-8")
    deciles.to_csv(output_path / "decile_metrics.csv", index=False)
    return report_path
