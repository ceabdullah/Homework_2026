"""Train one model. Every artifact of a run lands in results/<run_id>/.

run_id = timestamp + model + seed + config hash. Nothing is ever
overwritten, so every number in the thesis traces back to one directory.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.datasets import LesionDataset, class_weights
from src.models import build_model


def make_run_id(cfg: dict) -> str:
    h = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    return f"{time.strftime('%Y%m%d-%H%M%S')}_{cfg['model']}_s{cfg['seed']}_{h}"


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(size: int, train: bool):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    if not train:
        return A.Compose([A.Resize(size, size), norm, ToTensorV2()])

    # albumentations changed RandomResizedCrop's signature in 2.0 (size=(h,w))
    # and deprecated ShiftScaleRotate in favour of Affine. Support both so a
    # Colab image with a different pin still runs.
    try:
        crop = A.RandomResizedCrop(size=(size, size), scale=(0.7, 1.0))
    except TypeError:
        crop = A.RandomResizedCrop(size, size, scale=(0.7, 1.0))
    try:
        affine = A.Affine(translate_percent=(-0.0625, 0.0625), scale=(0.9, 1.1),
                          rotate=(-180, 180), p=0.7)
    except (TypeError, AttributeError):
        affine = A.ShiftScaleRotate(rotate_limit=180, p=0.7)
    return A.Compose([
        crop,
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        affine,
        A.ColorJitter(0.2, 0.2, 0.2, 0.05, p=0.5),
        norm, ToTensorV2(),
    ])


@torch.no_grad()
def evaluate(model, loader, device, classes):
    """Metrics over the model's FULL label space.

    `labels=range(len(classes))` is load-bearing: on the cross-domain target
    some source classes are absent, and without it sklearn silently averages
    over the present classes only, which makes in-domain and cross-domain
    macro-F1 incomparable.
    """
    model.eval()
    ys, ps, probs = [], [], []
    for x, y, _ in loader:
        out = model(x.to(device))
        probs.append(torch.softmax(out.float(), 1).cpu().numpy())
        ps.append(out.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    pr = np.concatenate(probs)
    idx = list(range(len(classes)))
    present = sorted(set(y.tolist()))
    return {
        "macro_f1": float(f1_score(y, p, average="macro", labels=idx, zero_division=0)),
        "macro_f1_present_only": float(
            f1_score(y, p, average="macro", labels=present, zero_division=0)),
        "n": int(len(y)),
        "classes_present": [classes[i] for i in present],
        "per_class": classification_report(y, p, labels=idx, target_names=classes,
                                           output_dict=True, zero_division=0),
        "confusion": confusion_matrix(y, p, labels=idx).tolist(),
    }, y, p, pr


def main(cfg_path: str, overrides: dict) -> str:
    cfg = yaml.safe_load(open(cfg_path))
    cfg.update(overrides)
    set_seed(cfg["seed"])
    use_cuda = torch.cuda.is_available() and not cfg.get("force_cpu", False)
    device = "cuda" if use_cuda else "cpu"
    classes = cfg["classes"]

    run_id = make_run_id(cfg)
    out = Path(cfg.get("results_dir", "results")) / run_id
    out.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(out / "config.yaml", "w"))
    json.dump({"device": device, "torch": torch.__version__,
               "platform": platform.platform(),
               "gpu": torch.cuda.get_device_name(0) if use_cuda else None},
              open(out / "environment.json", "w"), indent=2)

    tr_tf = build_transforms(cfg["image_size"], True)
    ev_tf = build_transforms(cfg["image_size"], False)

    def mk(split, tf):
        return LesionDataset(cfg["split_csv"], split, classes, tf, cfg["color_mode"])

    nw = cfg["num_workers"]
    train_ds = mk("train", tr_tf)
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                          num_workers=nw, pin_memory=use_cuda,
                          drop_last=len(train_ds) > cfg["batch_size"])
    val_dl = DataLoader(mk("val", ev_tf), batch_size=cfg["batch_size"] * 2, num_workers=nw)
    test_dl = DataLoader(mk("test", ev_tf), batch_size=cfg["batch_size"] * 2, num_workers=nw)

    model = build_model(cfg["model"], len(classes), pretrained=cfg.get("pretrained", True)).to(device)

    if cfg["imbalance"] == "class_weighted":
        w = class_weights(cfg["split_csv"], "train", classes).to(device)
        crit = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.get("label_smoothing", 0.0))
    else:
        crit = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    amp = use_cuda and cfg.get("amp", True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best, best_ep, history = -1.0, -1, []
    for ep in range(cfg["epochs"]):
        model.train()
        running, nb = 0.0, 0
        for x, y, _ in tqdm(train_dl, desc=f"ep{ep}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.detach())
            nb += 1
        sched.step()
        vm, *_ = evaluate(model, val_dl, device, classes)
        history.append({"epoch": ep, "train_loss": running / max(nb, 1),
                        "val_macro_f1": vm["macro_f1"]})
        print(f"ep{ep} loss={running / max(nb, 1):.4f} val_macro_f1={vm['macro_f1']:.4f}")
        if vm["macro_f1"] > best:
            best, best_ep = vm["macro_f1"], ep
            torch.save(model.state_dict(), out / "best.pt")

    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True))
    tm, y, p, pr = evaluate(model, test_dl, device, classes)
    np.savez(out / "test_predictions.npz", y=y, p=p, probs=pr)
    json.dump({"run_id": run_id, "best_val_macro_f1": best, "best_epoch": best_ep,
               "test": tm, "history": history},
              open(out / "metrics.json", "w"), indent=2)
    print(f"[done] {run_id}  test_macro_f1={tm['macro_f1']:.4f}")
    return run_id


def cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--model")
    ap.add_argument("--color_mode")
    ap.add_argument("--imbalance")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--image_size", type=int)
    ap.add_argument("--split_csv")
    ap.add_argument("--results_dir")
    ap.add_argument("--num_workers", type=int)
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=None,
                    help="random init; required on machines without HF hub access")
    ap.add_argument("--force-cpu", dest="force_cpu", action="store_true", default=None)
    a = ap.parse_args()
    main(a.config, {k: v for k, v in vars(a).items() if k != "config" and v is not None})


if __name__ == "__main__":
    cli()
