"""Phase 3: evaluate a HAM10000-trained model on PAD-UFES-20 and measure
how much target-domain data it takes to close the gap.

Three modes:
  zero_shot   source model, target test set, no adaptation. The headline.
  ablation    same, under each colour transform, to ask what the gap is made of.
  curve       fine-tune on an increasing fraction of target lesions.
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data.datasets import LesionDataset
from src.models import build_model
from src.preprocessing import COLOR_MODES
from src.train import build_transforms, evaluate, set_seed


def device_of(cfg: dict) -> str:
    return "cuda" if torch.cuda.is_available() and not cfg.get("force_cpu") else "cpu"


def load_run(run_id: str, results_dir: str = "results"):
    """Rebuild the architecture from the run's own config and load its best
    checkpoint. The config is read from disk, never re-specified, so a run
    can never be evaluated under a label space it was not trained on."""
    out = Path(results_dir) / run_id
    cfg = yaml.safe_load(open(out / "config.yaml"))
    dev = device_of(cfg)
    model = build_model(cfg["model"], len(cfg["classes"]), pretrained=False)
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev, weights_only=True))
    return model.to(dev), cfg, dev


def _loader(df: pd.DataFrame, cfg: dict, train: bool, color_mode: str,
            tmpdir: Path, batch_size: int):
    """LesionDataset reads a CSV, so a subset is materialised to a temp file.
    Keeps one code path for reading data -- the alternative is a second,
    subtly different one."""
    tmp = tmpdir / f"subset_{abs(hash(tuple(df['image_id'][:50]))) % 10**8}_{len(df)}.csv"
    df.to_csv(tmp, index=False)
    tf = build_transforms(cfg["image_size"], train)
    ds = LesionDataset(tmp, None, cfg["classes"], tf, color_mode)
    return DataLoader(ds, batch_size=batch_size, shuffle=train,
                      num_workers=cfg.get("num_workers", 4),
                      drop_last=train and len(ds) > batch_size)


def zero_shot(run_id: str, target_csv: str, color_mode: str | None = None,
              split: str | None = "test", results_dir: str = "results"):
    model, cfg, dev = load_run(run_id, results_dir)
    cm = color_mode or cfg["color_mode"]
    tf = build_transforms(cfg["image_size"], False)
    ds = LesionDataset(target_csv, split, cfg["classes"], tf, cm)
    dl = DataLoader(ds, batch_size=64, num_workers=cfg.get("num_workers", 4))
    m, y, p, pr = evaluate(model, dl, dev, cfg["classes"])
    return m, y, p, pr, ds.df


def finetune(model, sub_df, cfg, dev, seed, tmpdir, epochs=10, lr=1e-4):
    dl = _loader(sub_df, cfg, True, cfg["color_mode"], tmpdir, batch_size=16)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.1))
    model.train()
    for _ in range(epochs):
        for x, y, _ in dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            crit(model(x), y).backward()
            opt.step()
    return model


def few_shot_curve(run_id: str, target_csv: str,
                   fractions=(0.0, 0.05, 0.1, 0.25, 0.5, 1.0),
                   seeds=(42, 43, 44), epochs=10, lr=1e-4,
                   results_dir: str = "results") -> pd.DataFrame:
    """How much target data recovers the gap?

    Sampling is by LESION, not by image: sampling images would put other
    photographs of a training lesion into the target test set, which is the
    same leak Phase 1 guards against, and would make the curve rise for the
    wrong reason.
    """
    df = pd.read_csv(target_csv)
    test = df[df["split"] == "test"]
    pool = df[df["split"] == "train"]
    rows = []
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        for frac in fractions:
            for seed in seeds:
                set_seed(seed)
                model, cfg, dev = load_run(run_id, results_dir)
                n_used, n_lesions = 0, 0
                if frac > 0:
                    groups = pool["group"].unique()
                    rng = np.random.RandomState(seed)
                    keep = rng.choice(groups, max(1, int(round(len(groups) * frac))),
                                      replace=False)
                    sub = pool[pool["group"].isin(keep)]
                    model = finetune(model, sub, cfg, dev, seed, tmpdir, epochs, lr)
                    n_used, n_lesions = len(sub), len(keep)
                dl = _loader(test, cfg, False, cfg["color_mode"], tmpdir, batch_size=64)
                m, *_ = evaluate(model, dl, dev, cfg["classes"])
                rows.append({"frac": frac, "seed": seed, "n_target_images": n_used,
                             "n_target_lesions": n_lesions, "macro_f1": m["macro_f1"]})
                print(rows[-1])
                if frac == 0.0:
                    break   # no adaptation -> deterministic, one seed is the answer
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--target", default="data/splits/pad_shared4.csv")
    ap.add_argument("--mode", choices=["zero_shot", "ablation", "curve"], default="zero_shot")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--split", default="test",
                    help="target split to score; 'all' uses every target row")
    ap.add_argument("--ft_epochs", type=int, default=10)
    ap.add_argument("--ft_lr", type=float, default=1e-4)
    a = ap.parse_args()

    out = Path(a.results_dir) / a.run_id / "cross_domain"
    out.mkdir(parents=True, exist_ok=True)
    split = None if a.split == "all" else a.split

    if a.mode == "zero_shot":
        m, y, p, pr, df = zero_shot(a.run_id, a.target, split=split, results_dir=a.results_dir)
        np.savez(out / "pad_zero_shot.npz", y=y, p=p, probs=pr,
                 image_id=df["image_id"].values)
        json.dump(m, open(out / "pad_zero_shot.json", "w"), indent=2)
        src = json.load(open(Path(a.results_dir) / a.run_id / "metrics.json"))
        print(f"in-domain (HAM test) macro_f1 = {src['test']['macro_f1']:.4f}")
        print(f"cross-domain (PAD)   macro_f1 = {m['macro_f1']:.4f}")
        print(f"gap                            = "
              f"{src['test']['macro_f1'] - m['macro_f1']:+.4f}")

    elif a.mode == "ablation":
        res = {}
        for cm in COLOR_MODES:
            m, *_ = zero_shot(a.run_id, a.target, color_mode=cm, split=split,
                              results_dir=a.results_dir)
            res[cm] = m["macro_f1"]
            print(f"{cm:>16}: {m['macro_f1']:.4f}")
        json.dump(res, open(out / "color_ablation.json", "w"), indent=2)

    elif a.mode == "curve":
        df = few_shot_curve(a.run_id, a.target, epochs=a.ft_epochs, lr=a.ft_lr,
                            results_dir=a.results_dir)
        df.to_csv(out / "few_shot_curve.csv", index=False)
        print(df.groupby("frac")["macro_f1"].agg(["mean", "std"]).to_string())


if __name__ == "__main__":
    main()
