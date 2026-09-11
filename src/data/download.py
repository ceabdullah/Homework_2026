"""Download HAM10000 and PAD-UFES-20. Idempotent -- safe to re-run.

Two sources are supported per dataset because the Kaggle mirrors are not
guaranteed to stay up:

  HAM10000       Kaggle  kmader/skin-cancer-mnist-ham10000
                 (canonical: Harvard Dataverse doi:10.7910/DVN/DBW86T)
  PAD-UFES-20    Mendeley Data doi:10.17632/zr7vgbcyr2.1  <- canonical
                 Kaggle mirror slug is UNVERIFIED, see PAD_KAGGLE below.

Whichever source is used, verify_* asserts the row counts and column names
before anything downstream touches the files. If an assertion fires, the
mirror differs from the published dataset: adapt src/data/build_splits.py
and record the change, do not coerce it silently.
"""
import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")

HAM_KAGGLE = "kmader/skin-cancer-mnist-ham10000"

# UNVERIFIED. The canonical source is Mendeley Data 10.17632/zr7vgbcyr2.1
# (direct zip: https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/zr7vgbcyr2-1.zip).
# Prefer --source mendeley unless you have confirmed this mirror's schema.
PAD_KAGGLE = "mahdavi1202/skin-cancer"
PAD_MENDELEY_ZIP = (
    "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/zr7vgbcyr2-1.zip"
)

# Published reference distributions. These are the contract with the paper;
# a mismatch means the mirror is not the dataset you think it is.
HAM_EXPECTED = {"nv": 6705, "mel": 1113, "bkl": 1099, "bcc": 514,
                "akiec": 327, "vasc": 142, "df": 115}
HAM_EXPECTED_ROWS = 10015

# Verified against the dataset authors' analysis notebook
# (labcin-ufes/PAD-UFES-20, analysis/pad-ufes-20-analysis.ipynb).
PAD_EXPECTED = {"BCC": 845, "ACK": 730, "NEV": 244, "SEK": 235, "SCC": 192, "MEL": 52}
PAD_EXPECTED_ROWS = 2298
PAD_EXPECTED_LESIONS = 1641
PAD_EXPECTED_PATIENTS = 1373

# Only what load_pad() actually indexes by name. Asserting on the full 26
# columns would reject a legitimate release over an optional field.
PAD_REQUIRED_COLS = ["patient_id", "lesion_id", "img_id", "diagnostic"]

# Present in the published release; reported when absent, never fatal.
PAD_OPTIONAL_COLS = ["age", "gender", "region", "biopsed", "fitspatrick",
                     "diameter_1", "diameter_2", "itch", "grew", "hurt",
                     "changed", "bleed", "elevation", "smoke", "drink",
                     "pesticide", "skin_cancer_history", "cancer_history",
                     "background_father", "background_mother"]


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _unzip_all(dest: Path) -> None:
    for zf in sorted(dest.glob("*.zip")):
        with zipfile.ZipFile(zf) as z:
            z.extractall(dest)
        zf.unlink()
    # PAD-UFES-20 ships images as imgs_part_1.zip .. imgs_part_3.zip inside
    # the outer archive.
    for zf in sorted(dest.rglob("*.zip")):
        with zipfile.ZipFile(zf) as z:
            z.extractall(zf.parent)
        zf.unlink()


def kaggle_download(slug: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        print(f"[skip] {dest} already populated")
        return
    _run(["kaggle", "datasets", "download", "-d", slug, "-p", str(dest)])
    _unzip_all(dest)


def url_download(url: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        print(f"[skip] {dest} already populated")
        return
    out = dest / "download.zip"
    _run(["curl", "-L", "--fail", "-o", str(out), url])
    _unzip_all(dest)


def find_one(root: Path, pattern: str) -> Path:
    hits = sorted(root.rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"no {pattern} under {root}")
    return hits[0]


def verify_ham(root: Path) -> None:
    meta = pd.read_csv(find_one(root, "HAM10000_metadata*"))
    missing = {"lesion_id", "image_id", "dx"} - set(meta.columns)
    assert not missing, f"HAM10000 metadata missing columns: {missing}"
    assert len(meta) == HAM_EXPECTED_ROWS, f"expected {HAM_EXPECTED_ROWS} rows, got {len(meta)}"
    counts = meta["dx"].value_counts().to_dict()
    assert counts == HAM_EXPECTED, f"class distribution mismatch:\n got {counts}\n want {HAM_EXPECTED}"
    n_img = sum(1 for _ in root.rglob("*.jpg"))
    assert n_img >= HAM_EXPECTED_ROWS, f"expected >={HAM_EXPECTED_ROWS} images, found {n_img}"
    print(f"[ok] HAM10000: {len(meta)} rows, {meta['lesion_id'].nunique()} lesions, {n_img} images")
    print(meta["dx"].value_counts().to_string())


def verify_pad(root: Path) -> None:
    """Assert the PAD mirror matches the published dataset.

    Reports rather than raises on count drift, because a Kaggle mirror may
    legitimately be a subset -- but the column names are non-negotiable,
    build_splits.py reads them by name.
    """
    meta = pd.read_csv(find_one(root, "metadata.csv"))
    print(f"[info] PAD metadata columns ({len(meta.columns)}): {list(meta.columns)}")
    missing = set(PAD_REQUIRED_COLS) - set(meta.columns)
    assert not missing, (
        f"PAD metadata missing required columns: {missing}. "
        f"This mirror has a different schema -- adapt load_pad() in "
        f"src/data/build_splits.py and record the change."
    )
    absent = [c for c in PAD_OPTIONAL_COLS if c not in meta.columns]
    if absent:
        print(f"[info] optional columns not in this release: {absent}")

    counts = meta["diagnostic"].value_counts().to_dict()
    n_img = sum(1 for _ in root.rglob("*.png")) + sum(1 for _ in root.rglob("*.jpg"))
    composite = (meta["patient_id"].astype(str) + "_" + meta["lesion_id"].astype(str)).nunique()
    print(f"[ok] PAD-UFES-20: {len(meta)} rows, {meta['patient_id'].nunique()} patients, "
          f"{meta['lesion_id'].nunique()} raw lesion_ids, {composite} patient+lesion "
          f"groups, {n_img} image files")
    print(pd.Series(counts).to_string())

    # The one fact the published docs never state. build_splits.py groups by
    # the composite key either way; this says which assumption holds, so the
    # choice is defensible rather than lucky.
    if meta["lesion_id"].nunique() < composite:
        print("[IMPORTANT] lesion_id is NOT unique across patients. Grouping by "
              "lesion_id alone would merge different patients' lesions and leak.")
    else:
        print("[ok] lesion_id appears globally unique in this release.")

    if len(meta) != PAD_EXPECTED_ROWS or counts != PAD_EXPECTED:
        print(f"[WARN] differs from the published PAD-UFES-20 "
              f"({PAD_EXPECTED_ROWS} rows, {PAD_EXPECTED}).")
        print("[WARN] Report this mirror and its distribution in the thesis.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pad-source", choices=["mendeley", "kaggle"], default="mendeley",
                    help="mendeley is canonical; the kaggle slug is unverified")
    ap.add_argument("--skip-ham", action="store_true")
    ap.add_argument("--skip-pad", action="store_true")
    a = ap.parse_args()

    if not a.skip_ham:
        kaggle_download(HAM_KAGGLE, RAW / "ham10000")
        verify_ham(RAW / "ham10000")
    if not a.skip_pad:
        if a.pad_source == "mendeley":
            url_download(PAD_MENDELEY_ZIP, RAW / "pad_ufes_20")
        else:
            kaggle_download(PAD_KAGGLE, RAW / "pad_ufes_20")
        verify_pad(RAW / "pad_ufes_20")
    print("\nSource used for PAD-UFES-20:", a.pad_source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
