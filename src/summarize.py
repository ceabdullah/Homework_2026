"""Summary tables over runs. Instruction 5: print one of these after every
batch of runs, so a stale checkpoint never quietly becomes a thesis number."""
import argparse
import json
from pathlib import Path

import pandas as pd


def collect(results_dir: str = "results") -> pd.DataFrame:
    rows = []
    for m_path in sorted(Path(results_dir).rglob("metrics.json")):
        run = m_path.parent
        try:
            m = json.load(open(m_path))
            import yaml
            cfg = yaml.safe_load(open(run / "config.yaml"))
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"[warn] unreadable run {run}, skipped")
            continue
        rec = {
            "run_id": m.get("run_id", run.name),
            "split_csv": Path(cfg["split_csv"]).name,
            "model": cfg["model"],
            "imbalance": cfg["imbalance"],
            "color_mode": cfg["color_mode"],
            "seed": cfg["seed"],
            "epochs": cfg["epochs"],
            "val_macro_f1": round(m["best_val_macro_f1"], 4),
            "test_macro_f1": round(m["test"]["macro_f1"], 4),
        }
        for cls, d in m["test"]["per_class"].items():
            if isinstance(d, dict) and cls in cfg["classes"]:
                rec[f"recall_{cls}"] = round(d["recall"], 3)
        zs = run / "cross_domain" / "pad_zero_shot.json"
        if zs.exists():
            rec["pad_macro_f1"] = round(json.load(open(zs))["macro_f1"], 4)
            rec["gap"] = round(rec["test_macro_f1"] - rec["pad_macro_f1"], 4)
        rows.append(rec)
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame, by=("split_csv", "model", "imbalance", "color_mode")) -> pd.DataFrame:
    """Mean +/- std over seeds. The std is the noise floor: no effect smaller
    than it may be claimed as a finding."""
    by = [c for c in by if c in df.columns]
    metrics = [c for c in ("test_macro_f1", "pad_macro_f1", "gap") if c in df.columns]
    g = df.groupby(by)[metrics].agg(["mean", "std", "count"]).round(4)
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--csv", help="also write the per-run table here")
    a = ap.parse_args()
    df = collect(a.results_dir)
    if df.empty:
        print(f"no runs under {a.results_dir}")
        return
    pd.set_option("display.width", 200, "display.max_columns", 50)
    print("=== per-run ===")
    print(df.to_string(index=False))
    print("\n=== mean +/- std over seeds ===")
    print(aggregate(df).to_string())
    if a.csv:
        df.to_csv(a.csv, index=False)
        print(f"\n[done] wrote {a.csv}")


if __name__ == "__main__":
    main()
