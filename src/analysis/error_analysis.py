"""Export the most confident cross-domain failures for manual labelling.

The hand labels are the qualitative layer of the thesis. Do not automate
the failure_category column -- a model's own explanation of why it failed
is not evidence.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CATEGORIES = ["color", "framing", "occlusion(hair)", "scale", "ambiguous",
              "label_noise", "other"]


def export_failures(run_id: str, target_csv: str, n: int = 100,
                    results_dir: str = "results") -> Path:
    run = Path(results_dir) / run_id
    d = np.load(run / "cross_domain" / "pad_zero_shot.npz", allow_pickle=True)
    y, p, probs, ids = d["y"], d["p"], d["probs"], d["image_id"]
    classes = yaml.safe_load(open(run / "config.yaml"))["classes"]

    wrong = np.where(y != p)[0]
    if len(wrong) == 0:
        print("no misclassifications -- check that the target labels loaded correctly")
        return run / "cross_domain" / "failures_to_label.csv"
    conf = probs[wrong, p[wrong]]
    order = wrong[np.argsort(-conf)][:n]     # most confidently wrong first

    meta = pd.read_csv(target_csv).drop_duplicates("image_id").set_index("image_id")
    out = pd.DataFrame({
        "image_id": ids[order],
        "true": [classes[i] for i in y[order]],
        "pred": [classes[i] for i in p[order]],
        "confidence": probs[order, p[order]].round(4),
        "path": [meta.loc[i, "path"] if i in meta.index else "" for i in ids[order]],
        "failure_category": "",     # one of CATEGORIES, filled by hand
        "notes": "",
    })
    dest = run / "cross_domain" / "failures_to_label.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"wrote {len(out)} rows to {dest}")
    print(f"Fill 'failure_category' by hand, one of: {CATEGORIES}")
    print("This is the qualitative contribution -- do not automate it.")
    return dest


def summarize_labels(labelled_csv: str) -> pd.DataFrame:
    """Run after the hand labelling. Feeds figure 5."""
    df = pd.read_csv(labelled_csv)
    done = df[df["failure_category"].astype(str).str.strip() != ""]
    print(f"{len(done)}/{len(df)} labelled")
    if done.empty:
        return done
    tab = pd.crosstab(done["failure_category"], done["true"], margins=True)
    print(tab.to_string())
    return tab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--target", default="data/splits/pad_shared4.csv")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--summarize", help="path to the hand-labelled CSV")
    a = ap.parse_args()
    if a.summarize:
        summarize_labels(a.summarize)
    else:
        export_failures(a.run_id, a.target, a.n, a.results_dir)


if __name__ == "__main__":
    main()
