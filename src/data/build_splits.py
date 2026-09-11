"""Build every subset and split used in the thesis.

Run ONCE. The CSVs it writes to data/splits/ are committed and are the
single source of truth for every run. Do not regenerate them to "fix" a
result -- regenerate only when the raw data or the mapping changes, and
say so in the lab notebook.

The one rule this file exists to enforce: HAM10000 contains several images
of the same lesion. Splitting by image puts near-duplicates of a training
lesion into the test set and inflates macro-F1 by several points. Every
split here is made at lesion level and checked by assert_no_leak().
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

RAW = Path("data/raw")
SPLITS = Path("data/splits")
SEED = 42

HAM_CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]

# ---------------------------------------------------------------------------
# Shared label space between HAM10000 and PAD-UFES-20.
# Every decision here is argued in the thesis chapter "Datasets and label
# mapping"; a reviewer will go through it line by line.
#
#   BCC -> bcc     basal cell carcinoma, same entity in both datasets.
#   MEL -> mel     melanoma, same entity.
#   NEV -> nv      melanocytic nevus, same entity.
#   ACK -> akiec   actinic keratosis. HAM's 'akiec' is actinic keratosis AND
#                  intraepithelial carcinoma (Bowen's); PAD's 'ACK' is
#                  actinic keratosis only. The mapping is therefore slightly
#                  broader on the HAM side. Documented, accepted.
#   SEK -> bkl     seborrheic keratosis maps into HAM's broader 'benign
#                  keratosis-like lesions'. OPTIONAL 5th class: enable with
#                  --include-sek to run the committee's sensitivity check.
#   SCC -> EXCLUDED. PAD's SCC is invasive squamous cell carcinoma (and
#                  includes Bowen's disease). Folding it into 'akiec' is
#                  defensible but conflates in-situ with invasive disease.
#                  Excluded to keep the comparison clean.
# ---------------------------------------------------------------------------
PAD_TO_HAM = {"BCC": "bcc", "MEL": "mel", "ACK": "akiec", "NEV": "nv"}
PAD_TO_HAM_SEK = {**PAD_TO_HAM, "SEK": "bkl"}

SHARED_CLASSES = ["bcc", "mel", "akiec", "nv"]
SHARED_CLASSES_SEK = ["bcc", "mel", "akiec", "nv", "bkl"]


def _find(root: Path, pattern: str) -> Path:
    hits = sorted(root.rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"no {pattern} under {root} -- run src/data/download.py first")
    return hits[0]


def load_ham(root: Path = RAW / "ham10000") -> pd.DataFrame:
    meta = pd.read_csv(_find(root, "HAM10000_metadata*"))
    paths = {p.stem: p for p in root.rglob("*.jpg")}
    missing = [i for i in meta["image_id"] if i not in paths]
    if missing:
        raise FileNotFoundError(f"{len(missing)} HAM images referenced by metadata not on disk, "
                                f"e.g. {missing[:3]}")
    meta["path"] = meta["image_id"].map(lambda i: str(paths[i]))
    meta = meta.rename(columns={"dx": "label", "lesion_id": "group"})
    meta["dataset"] = "ham10000"
    cols = ["image_id", "path", "label", "group", "dataset"]
    for extra in ("age", "sex", "localization"):
        if extra in meta.columns:
            cols.append(extra)
    return meta[cols].reset_index(drop=True)


def load_pad(root: Path = RAW / "pad_ufes_20", include_sek: bool = False,
             group_by: str = "lesion") -> pd.DataFrame:
    """PAD-UFES-20 loader.

    Schema verified against the dataset authors' own analysis notebook
    (labcin-ufes/PAD-UFES-20, analysis/pad-ufes-20-analysis.ipynb):

        patient_id, lesion_id, smoke, drink, background_father,
        background_mother, age, pesticide, gender, skin_cancer_history,
        diameter_1, diameter_2, diagnostic, itch, grew, hurt, changed,
        bleed, elevation, img_id, biopsed

    img_id carries the file extension ("PAT_1516_1765_530.png"), so it is
    matched against Path.name, not Path.stem. Images are .png.
    1373 patients / 1641 lesions / 2298 images.

    group_by:
      "lesion"  -- patient_id + lesion_id, the composite key. The published
                   documentation does NOT state whether lesion_id is unique
                   across patients or numbered within a patient. The
                   composite is identical to lesion_id if it is globally
                   unique, and correct if it is not; bare lesion_id would
                   silently merge different patients' lesions into one
                   group and hand you a leak-free-looking split that leaks.
      "patient" -- group every lesion of a patient together. Stricter: the
                   same skin, camera and lighting recur across a patient's
                   lesions. Use it as a sensitivity check.

    If a mirror differs, adapt HERE and report the change -- do not
    silently rename columns elsewhere.
    """
    meta = pd.read_csv(_find(root, "metadata.csv"))
    required = {"img_id", "diagnostic", "lesion_id", "patient_id"}
    missing = required - set(meta.columns)
    if missing:
        raise KeyError(f"PAD metadata is missing {missing}; found {list(meta.columns)}. "
                       f"Adapt load_pad() and record the deviation.")

    paths: dict[str, Path] = {}
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"):
        paths.update({p.name: p for p in root.rglob(ext)})
    meta["path"] = meta["img_id"].map(lambda i: str(paths[i]) if i in paths else None)
    n_no_file = meta["path"].isna().sum()
    if n_no_file:
        print(f"[warn] {n_no_file} PAD rows have no image file on disk, dropped")
    meta = meta.dropna(subset=["path"])

    mapping = PAD_TO_HAM_SEK if include_sek else PAD_TO_HAM
    meta["label"] = meta["diagnostic"].map(mapping)
    dropped = meta["label"].isna().sum()
    if dropped:
        kept = meta.loc[meta["label"].isna(), "diagnostic"].value_counts().to_dict()
        print(f"[info] {dropped} PAD rows dropped (unmapped diagnoses): {kept}")
    meta = meta.dropna(subset=["label"])

    # PAD photographs a lesion several times, same hazard as HAM10000.
    pid = meta["patient_id"].astype(str)
    lid = meta["lesion_id"].astype(str)
    if group_by == "patient":
        meta["group"] = pid
    elif group_by == "lesion":
        meta["group"] = pid + "_" + lid
        # Diagnostic, not an assertion: says which assumption the data
        # actually supports, so the choice can be defended in the write-up.
        if lid.nunique() < meta["group"].nunique():
            print(f"[info] PAD lesion_id is NOT unique across patients "
                  f"({lid.nunique()} raw ids vs {meta['group'].nunique()} "
                  f"patient+lesion groups) -- grouping by the composite key")
    else:
        raise ValueError(f"group_by must be 'lesion' or 'patient', got {group_by!r}")
    meta["dataset"] = "pad_ufes_20"
    meta = meta.rename(columns={"img_id": "image_id"})
    cols = ["image_id", "path", "label", "group", "dataset"]
    for extra in ("age", "region", "biopsed", "patient_id"):
        if extra in meta.columns:
            cols.append(extra)
    return meta[cols].reset_index(drop=True)


def grouped_split(df: pd.DataFrame, seed: int = SEED, n_splits: int = 7) -> pd.DataFrame:
    """Stratified split that keeps every image of one lesion in one fold.

    fold 0 -> test, fold 1 -> val, remainder -> train  (roughly 14/14/72).
    """
    df = df.reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [te for _, te in sgkf.split(df, df["label"], groups=df["group"])]
    test_idx, val_idx = folds[0], folds[1]
    out = df.copy()
    out["split"] = "train"
    col = out.columns.get_loc("split")
    out.iloc[val_idx, col] = "val"
    out.iloc[test_idx, col] = "test"
    return out


def assert_no_leak(df: pd.DataFrame, name: str = "") -> None:
    per_group = df.groupby("group")["split"].nunique()
    bad = per_group[per_group > 1]
    assert bad.empty, (f"LEAK in {name}: {len(bad)} lesions span multiple splits, "
                       f"e.g. {list(bad.index[:5])}")
    dup_paths = df["path"].duplicated().sum()
    assert dup_paths == 0, f"LEAK in {name}: {dup_paths} duplicate image paths"
    print(f"[ok] {name}: no lesion-level leakage ({df['group'].nunique()} lesions)")


def check_class_coverage(df: pd.DataFrame, name: str, min_test: int = 5) -> None:
    """Warn when a class is missing or nearly missing from a split.

    A class with zero test images makes macro-F1 incomparable across
    datasets: it contributes a hard zero on one side and nothing on the
    other. PAD-UFES-20 has only 52 MEL images, so this is a live hazard,
    not a hypothetical. If it fires, re-split with fewer folds (a larger
    test fraction) or merge the class -- and say which in the write-up.
    """
    tab = pd.crosstab(df["label"], df["split"])
    for split in ("train", "val", "test"):
        if split not in tab.columns:
            print(f"[WARN] {name}: split {split!r} is empty")
            continue
        thin = tab[tab[split] < (min_test if split == "test" else 1)][split]
        for cls, n in thin.items():
            print(f"[WARN] {name}: class {cls!r} has only {n} images in {split!r}"
                  f" -- macro-F1 over this split is unstable for it")


def build_balanced(ham: pd.DataFrame, n_per_class: int = 115, seed: int = SEED) -> pd.DataFrame:
    """Downsample every class to the size of the smallest (df, 115 images).

    Sampling is at lesion level so the balanced set is leak-free by
    construction. A class can land a little under n_per_class when its
    remaining lesions are multi-image; the realized counts are printed.
    """
    rng = np.random.RandomState(seed)
    parts = []
    for cls in HAM_CLASSES:
        sub = ham[ham["label"] == cls]
        groups = sub["group"].unique().copy()
        rng.shuffle(groups)
        by_group = {g: rows for g, rows in sub.groupby("group")}
        picked, count = [], 0
        for g in groups:
            rows = by_group[g]
            if count + len(rows) > n_per_class:
                continue          # try a smaller lesion instead of overshooting
            picked.append(rows)
            count += len(rows)
            if count == n_per_class:
                break
        parts.append(pd.concat(picked))
    out = pd.concat(parts).reset_index(drop=True)
    print("[info] balanced subset realized counts:")
    print(out["label"].value_counts().to_string())
    return out


def report(name: str, d: pd.DataFrame) -> None:
    print(f"\n--- {name}  ({len(d)} images, {d['group'].nunique()} lesions) ---")
    print(pd.crosstab(d["label"], d["split"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ham-root", default=str(RAW / "ham10000"))
    ap.add_argument("--pad-root", default=str(RAW / "pad_ufes_20"))
    ap.add_argument("--out", default=str(SPLITS))
    ap.add_argument("--include-sek", action="store_true",
                    help="add PAD SEK -> HAM bkl as a 5th shared class (sensitivity check)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--pad-group-by", choices=["lesion", "patient"], default="lesion",
                    help="'patient' is the stricter sensitivity check")
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    shared = SHARED_CLASSES_SEK if a.include_sek else SHARED_CLASSES
    suffix = "5" if a.include_sek else "4"

    ham = load_ham(Path(a.ham_root))
    pad = load_pad(Path(a.pad_root), include_sek=a.include_sek,
                   group_by=a.pad_group_by)

    # Phase 1 -- balanced pilot.
    bal = grouped_split(build_balanced(ham, seed=a.seed), seed=a.seed)
    assert_no_leak(bal, "ham_balanced")
    check_class_coverage(bal, "ham_balanced")
    bal.to_csv(out_dir / "ham_balanced.csv", index=False)

    # Phase 2 -- full HAM10000.
    full = grouped_split(ham, seed=a.seed)
    assert_no_leak(full, "ham_full")
    check_class_coverage(full, "ham_full")
    full.to_csv(out_dir / "ham_full.csv", index=False)

    # Phase 3 -- shared label space, source and target.
    # Re-split the source subset so train/val/test stay balanced after the
    # class filter (filtering a 7-class split leaves uneven fold sizes).
    src = grouped_split(full[full["label"].isin(shared)].drop(columns="split"), seed=a.seed)
    assert_no_leak(src, f"ham_shared{suffix}")
    check_class_coverage(src, f"ham_shared{suffix}")
    src.to_csv(out_dir / f"ham_shared{suffix}.csv", index=False)

    tgt = grouped_split(pad, seed=a.seed)
    assert_no_leak(tgt, f"pad_shared{suffix}")
    check_class_coverage(tgt, f"pad_shared{suffix}")
    tgt.to_csv(out_dir / f"pad_shared{suffix}.csv", index=False)

    for name, d in [("ham_balanced", bal), ("ham_full", full),
                    (f"ham_shared{suffix}", src), (f"pad_shared{suffix}", tgt)]:
        report(name, d)
    print(f"\n[done] wrote splits to {out_dir}")


if __name__ == "__main__":
    main()
