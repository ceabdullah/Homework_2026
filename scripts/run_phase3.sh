#!/usr/bin/env bash
# Phase 3 -- cross-domain. The contribution; everything before it is setup.
#
# Note on the colour ablation. `cross_domain.py --mode ablation` transforms
# the TARGET at evaluation time only, against a model trained on raw images.
# That is cheap but confounded: the drop it shows mixes the domain effect
# with a train/test preprocessing mismatch. The honest version trains a
# matched source model per colour mode, which is what step 3 below does.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$PWD"

# 1. source baselines in the shared label space
for m in resnet50 convnext_tiny; do
  for s in 42 43 44; do
    python3 -m src.train --config configs/phase3_crossdomain.yaml --model "$m" --seed "$s"
  done
done

# 2. matched colour-mode source models (the uncounfounded ablation)
for cm in shades_of_gray grayscale; do
  for s in 42 43 44; do
    python3 -m src.train --config configs/phase3_crossdomain.yaml \
      --model resnet50 --color_mode "$cm" --seed "$s"
  done
done

# 3. zero-shot every source run onto PAD; summarize prints the gap column
python3 - <<'PY'
import subprocess, yaml
from pathlib import Path
for cfg in sorted(Path("results").glob("*/config.yaml")):
    c = yaml.safe_load(open(cfg))
    if "shared4" not in c["split_csv"]:
        continue
    subprocess.run(["python3", "-m", "src.cross_domain", "--run_id", cfg.parent.name,
                    "--mode", "zero_shot"], check=True)
PY

python3 -m src.summarize --csv results/phase3_summary.csv
echo
echo "Now pick the best source run_id and run:"
echo "  python3 -m src.cross_domain --run_id <RUN_ID> --mode curve"
echo "  python3 -m src.analysis.error_analysis --run_id <RUN_ID>"
echo "  python3 -m src.analysis.figures --run_id <RUN_ID>"
