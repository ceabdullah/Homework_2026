#!/usr/bin/env bash
# Phase 2 -- full HAM10000. 2 models x 2 imbalance strategies x 3 seeds = 12
# runs, roughly 25-40 min each on a 3080 Ti at 224px.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$PWD"
for m in resnet50 convnext_tiny; do
  for imb in none class_weighted; do
    for s in 42 43 44; do
      python3 -m src.train --config configs/phase2_full.yaml \
        --model "$m" --imbalance "$imb" --seed "$s"
    done
  done
done
python3 -m src.summarize --csv results/phase2_summary.csv
