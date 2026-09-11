"""Every figure in the thesis, generated from a script. You will regenerate
them at least five times; never touch them by hand.

    python -m src.analysis.figures --run_id <SRC_RUN> [--figures 1 2 3 4 5]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9})


def _save(fig, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest)
    fig.savefig(dest.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[fig] {dest}")


def fig1_class_distribution(splits_dir: Path, dest: Path) -> None:
    """Bar, log scale. The whole imbalance argument in one picture."""
    files = [("HAM10000 (full)", splits_dir / "ham_full.csv"),
             ("PAD-UFES-20", splits_dir / "pad_shared4.csv")]
    files = [(n, p) for n, p in files if p.exists()]
    if not files:
        print("[skip] fig1: no split CSVs")
        return
    fig, axes = plt.subplots(1, len(files), figsize=(4.2 * len(files), 3))
    axes = np.atleast_1d(axes)
    for ax, (name, path) in zip(axes, files):
        vc = pd.read_csv(path)["label"].value_counts()
        ax.bar(vc.index, vc.values, color="#4C72B0")
        ax.set_yscale("log")
        ax.set_title(f"{name}  (n={vc.sum()})")
        ax.set_ylabel("images (log)")
        for i, v in enumerate(vc.values):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=7)
    _save(fig, dest)


def fig2_confusion(run: Path, dest: Path) -> None:
    """In-domain vs cross-domain, side by side, row-normalised."""
    classes = yaml.safe_load(open(run / "config.yaml"))["classes"]
    panels = [("In-domain (HAM test)",
               np.array(json.load(open(run / "metrics.json"))["test"]["confusion"]))]
    zs = run / "cross_domain" / "pad_zero_shot.json"
    if zs.exists():
        panels.append(("Cross-domain (PAD, zero-shot)",
                       np.array(json.load(open(zs))["confusion"])))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6))
    for ax, (title, cm) in zip(np.atleast_1d(axes), panels):
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, dest)


def fig3_color_ablation(run: Path, dest: Path) -> None:
    src = run / "cross_domain" / "color_ablation.json"
    if not src.exists():
        print("[skip] fig3: run --mode ablation first")
        return
    res = json.load(open(src))
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(list(res), list(res.values()), color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylabel("macro-F1 on PAD (zero-shot)")
    ax.set_title("Colour ablation")
    for i, v in enumerate(res.values()):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    _save(fig, dest)


def fig4_few_shot_curve(run: Path, dest: Path) -> None:
    """The single most quotable result: macro-F1 vs target images used."""
    src = run / "cross_domain" / "few_shot_curve.csv"
    if not src.exists():
        print("[skip] fig4: run --mode curve first")
        return
    df = pd.read_csv(src)
    g = df.groupby("n_target_images")["macro_f1"].agg(["mean", "std"]).fillna(0).reset_index()
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(g["n_target_images"], g["mean"], "o-", color="#4C72B0")
    ax.fill_between(g["n_target_images"], g["mean"] - g["std"], g["mean"] + g["std"],
                    alpha=0.2, color="#4C72B0")
    in_domain = json.load(open(run / "metrics.json"))["test"]["macro_f1"]
    ax.axhline(in_domain, ls="--", c="grey", lw=1)
    ax.text(g["n_target_images"].max(), in_domain, " in-domain ceiling",
            va="bottom", ha="right", fontsize=7, color="grey")
    ax.set_xlabel("labelled target-domain images used for fine-tuning")
    ax.set_ylabel("macro-F1 on PAD test")
    ax.set_title("Few-shot recovery")
    _save(fig, dest)


def fig5_failure_categories(run: Path, dest: Path) -> None:
    src = run / "cross_domain" / "failures_to_label.csv"
    if not src.exists():
        print("[skip] fig5: export failures first")
        return
    df = pd.read_csv(src)
    df = df[df["failure_category"].astype(str).str.strip() != ""]
    if df.empty:
        print("[skip] fig5: failure_category is still empty -- label by hand")
        return
    vc = df["failure_category"].value_counts()
    fig, ax = plt.subplots(figsize=(4.4, 3))
    ax.barh(vc.index[::-1], vc.values[::-1], color="#C44E52")
    ax.set_xlabel(f"count (of {len(df)} hand-labelled failures)")
    ax.set_title("Cross-domain failure categories")
    _save(fig, dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--splits_dir", default="data/splits")
    ap.add_argument("--figures", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    a = ap.parse_args()
    run = Path(a.results_dir) / a.run_id
    out = run / "figures"
    jobs = {
        1: lambda: fig1_class_distribution(Path(a.splits_dir), out / "fig1_class_distribution.png"),
        2: lambda: fig2_confusion(run, out / "fig2_confusion.png"),
        3: lambda: fig3_color_ablation(run, out / "fig3_color_ablation.png"),
        4: lambda: fig4_few_shot_curve(run, out / "fig4_few_shot_curve.png"),
        5: lambda: fig5_failure_categories(run, out / "fig5_failure_categories.png"),
    }
    for k in a.figures:
        jobs[k]()


if __name__ == "__main__":
    main()
