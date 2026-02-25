from __future__ import annotations

import argparse
import json

from .pipeline import run_training_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Behavioral health outreach propensity modeling")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="artifacts")
    return parser


def run_train() -> None:
    args = _build_parser().parse_args()
    summary = run_training_pipeline(args.config, args.output)
    print(json.dumps(summary.__dict__, indent=2))


def run_report() -> None:
    run_train()


if __name__ == "__main__":
    run_train()
