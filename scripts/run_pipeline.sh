#!/usr/bin/env bash
set -euo pipefail
python -m outreach_model.cli --config configs/default.yaml --output artifacts
