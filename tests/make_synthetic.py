"""Generate fake HAM10000 / PAD-UFES-20 trees with the real schema.

Not data -- a fixture. It exists so the pipeline (splits, leakage check,
training loop, cross-domain eval, figures) can be verified on a machine
with no GPU and no dataset access, before a minute of real compute is spent.

The fixture reproduces the two properties that break naive pipelines:
  * multiple images per lesion  -> the leakage hazard
  * heavy class imbalance       -> the macro-F1 argument
and gives the two domains different colour statistics, so a cross-domain
gap is present and measurable.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HAM_CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
HAM_SHARE = {"nv": 0.45, "mel": 0.12, "bkl": 0.12, "bcc": 0.11,
             "akiec": 0.09, "vasc": 0.06, "df": 0.05}
PAD_CLASSES = ["BCC", "ACK", "NEV", "SEK", "SCC", "MEL"]
PAD_SHARE = {"BCC": 0.37, "ACK": 0.32, "NEV": 0.11, "SEK": 0.10, "SCC": 0.08, "MEL": 0.02}

# Per-class hue centres, shared by both domains, so a class is learnable.
CLASS_HUE = {"nv": 20, "mel": 5, "bkl": 40, "bcc": 60, "akiec": 80, "vasc": 120, "df": 150}
PAD_TO_HAM_HUE = {"BCC": "bcc", "ACK": "akiec", "NEV": "nv", "SEK": "bkl",
                  "SCC": "akiec", "MEL": "mel"}


def _lesion_image(rng, hue, size, domain_shift):
    """Coloured blob on skin-toned background, plus a domain-specific
    illumination cast so the two fixtures are genuinely different domains."""
    base = np.zeros((size, size, 3), np.float32)
    base[..., 0] = 190 + rng.normal(0, 6)
    base[..., 1] = 150 + rng.normal(0, 6)
    base[..., 2] = 130 + rng.normal(0, 6)
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = rng.uniform(0.35, 0.65, 2) * size
    r = rng.uniform(0.15, 0.3) * size
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) < r ** 2
    lesion = np.array([hue + 60, hue + 20, hue], np.float32)
    base[mask] = lesion + rng.normal(0, 8, 3)
    base += rng.normal(0, 4, base.shape)
    base *= np.array(domain_shift, np.float32)      # illumination cast
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def build_ham(root: Path, n_lesions: int, size: int, seed: int) -> None:
    rng = np.random.RandomState(seed)
    img_dir = root / "HAM10000_images_part_1"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for li in range(n_lesions):
        cls = rng.choice(HAM_CLASSES, p=[HAM_SHARE[c] for c in HAM_CLASSES])
        lesion_id = f"HAM_{li:07d}"
        # ~30% of lesions are photographed more than once: the leak hazard.
        for k in range(1 if rng.rand() > 0.3 else rng.randint(2, 4)):
            image_id = f"ISIC_{li:07d}_{k}"
            _lesion_image(rng, CLASS_HUE[cls], size, (1.0, 1.0, 1.0)).save(
                img_dir / f"{image_id}.jpg", quality=92)
            rows.append({"lesion_id": lesion_id, "image_id": image_id, "dx": cls,
                         "dx_type": "histo", "age": float(rng.randint(20, 85)),
                         "sex": rng.choice(["male", "female"]),
                         "localization": rng.choice(["back", "face", "trunk", "arm"])})
    pd.DataFrame(rows).to_csv(root / "HAM10000_metadata.csv", index=False)
    print(f"[fixture] HAM: {len(rows)} images / {n_lesions} lesions -> {root}")


def build_pad(root: Path, n_lesions: int, size: int, seed: int) -> None:
    rng = np.random.RandomState(seed + 1)
    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for li in range(n_lesions):
        dx = rng.choice(PAD_CLASSES, p=[PAD_SHARE[c] for c in PAD_CLASSES])
        hue = CLASS_HUE[PAD_TO_HAM_HUE[dx]]
        patient_id = f"PAT_{li // 2}"
        lesion_id = f"{li}"
        for k in range(1 if rng.rand() > 0.4 else rng.randint(2, 4)):
            img_id = f"PAT_{li // 2}_{li}_{k}.png"
            # Warm smartphone cast + lower effective sharpness: the domain gap.
            _lesion_image(rng, hue, size, (1.12, 0.98, 0.88)).save(img_dir / img_id)
            rows.append({"patient_id": patient_id, "lesion_id": lesion_id, "img_id": img_id,
                         "diagnostic": dx, "age": int(rng.randint(20, 90)),
                         "region": rng.choice(["FACE", "ARM", "BACK", "CHEST"]),
                         "itch": rng.choice(["True", "False"]),
                         "biopsed": bool(rng.rand() > 0.3)})
    pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)
    print(f"[fixture] PAD: {len(rows)} images / {n_lesions} lesions -> {root}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--ham-lesions", type=int, default=260)
    ap.add_argument("--pad-lesions", type=int, default=160)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    build_ham(out / "ham10000", a.ham_lesions, a.size, a.seed)
    build_pad(out / "pad_ufes_20", a.pad_lesions, a.size, a.seed)


if __name__ == "__main__":
    main()
