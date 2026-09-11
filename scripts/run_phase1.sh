#!/usr/bin/env bash
# Phase 1 -- balanced pilot, three seeds. The number that matters here is
# the STD of test macro-F1 across seeds: the noise floor. No effect claimed
# later may be smaller than it.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$PWD"
for s in 42 43 44; do
  python3 -m src.train --config configs/phase1_balanced.yaml --seed "$s"
done
python3 -m src.summarize --csv results/phase1_summary.csv
