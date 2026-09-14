"""Zero-shot every shared-label-space source run onto the target domain.

Phase 3 step 3. Idempotent: a run whose cross_domain/pad_zero_shot.json
already exists is skipped, so re-running after adding seeds is cheap.

    python -m src.zero_shot_all [--target data/splits/pad_shared4.csv]
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--target", default="data/splits/pad_shared4.csv")
    ap.add_argument("--match", default="shared4",
                    help="only runs whose split_csv contains this substring")
    ap.add_argument("--force", action="store_true", help="redo runs already scored")
    a = ap.parse_args()

    if not Path(a.target).exists():
        print(f"[FAIL] target split not found: {a.target}")
        print("       run `python -m src.data.build_splits` first")
        return 1

    done, skipped, failed = 0, 0, []
    for cfg_path in sorted(Path(a.results_dir).glob("*/config.yaml")):
        cfg = yaml.safe_load(open(cfg_path))
        if a.match not in str(cfg.get("split_csv", "")):
            continue
        run_id = cfg_path.parent.name
        marker = cfg_path.parent / "cross_domain" / "pad_zero_shot.json"
        if marker.exists() and not a.force:
            print(f"[skip] {run_id} already scored")
            skipped += 1
            continue
        print(f"[run ] {run_id}")
        r = subprocess.run([sys.executable, "-m", "src.cross_domain",
                            "--run_id", run_id, "--results_dir", a.results_dir,
                            "--target", a.target, "--mode", "zero_shot"])
        if r.returncode != 0:
            failed.append(run_id)
        else:
            done += 1

    print(f"\n[done] scored {done}, skipped {skipped}, failed {len(failed)}")
    if failed:
        print("failed runs:", ", ".join(failed))
        return 1
    if done == 0 and skipped == 0:
        print(f"no runs under {a.results_dir} matched {a.match!r} -- "
              f"train a source model first (scripts/run_phase3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
