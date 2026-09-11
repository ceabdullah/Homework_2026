#!/usr/bin/env bash
# End-to-end pipeline check on synthetic fixtures. No GPU, no dataset, no
# network. Proves every stage runs and wires together before real compute is
# spent. Takes a few minutes on CPU.
#
#   bash tests/smoke_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$PWD"
export NO_ALBUMENTATIONS_UPDATE=1

SYN=data/synthetic
SPL=data/splits_smoke
RES=results/smoke

echo "=== 0. unit tests ==="
python3 tests/test_splits.py | tail -1

echo "=== 1. synthetic fixtures ==="
python3 tests/make_synthetic.py --out "$SYN"

echo "=== 2. splits (incl. leakage assertions) ==="
python3 -m src.data.build_splits \
  --ham-root "$SYN/ham10000" --pad-root "$SYN/pad_ufes_20" --out "$SPL"

echo "=== 3. train: 7-class pilot, 2 seeds ==="
for s in 42 43; do
  python3 -m src.train --config configs/smoke.yaml --seed "$s"
done

echo "=== 4. train: shared-4 source model ==="
python3 -m src.train --config configs/smoke_shared4.yaml --seed 42
SRC=$(ls -t "$RES" | head -1)
echo "source run_id = $SRC"

echo "=== 5. cross-domain: zero-shot / ablation / curve ==="
python3 -m src.cross_domain --run_id "$SRC" --results_dir "$RES" \
  --target "$SPL/pad_shared4.csv" --mode zero_shot
python3 -m src.cross_domain --run_id "$SRC" --results_dir "$RES" \
  --target "$SPL/pad_shared4.csv" --mode ablation
python3 -m src.cross_domain --run_id "$SRC" --results_dir "$RES" \
  --target "$SPL/pad_shared4.csv" --mode curve --ft_epochs 2

echo "=== 6. error analysis + figures ==="
python3 -m src.analysis.error_analysis --run_id "$SRC" --results_dir "$RES" \
  --target "$SPL/pad_shared4.csv" --n 20
python3 -m src.analysis.figures --run_id "$SRC" --results_dir "$RES" --splits_dir "$SPL"

echo "=== 7. summary table ==="
python3 -m src.summarize --results_dir "$RES"

echo
echo "SMOKE TEST PASSED"
