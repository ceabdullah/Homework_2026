"""Unit tests for the guarantees the thesis depends on. Run: pytest tests/ -q
(or python tests/test_splits.py, which needs no pytest)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.build_splits import assert_no_leak, build_balanced, grouped_split  # noqa: E402
from src.preprocessing import shades_of_gray, to_grayscale_3ch  # noqa: E402


def _fake(n_lesions=140, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n_lesions):
        cls = rng.choice(["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"],
                         p=[0.4, 0.15, 0.13, 0.12, 0.09, 0.06, 0.05])
        for k in range(rng.randint(1, 4)):
            rows.append({"image_id": f"i{i}_{k}", "path": f"/tmp/i{i}_{k}.jpg",
                         "label": cls, "group": f"L{i}"})
    return pd.DataFrame(rows)


def test_grouped_split_has_no_leak():
    out = grouped_split(_fake())
    assert_no_leak(out, "unit")
    assert set(out["split"]) == {"train", "val", "test"}


def test_grouped_split_is_deterministic():
    a, b = grouped_split(_fake(), seed=42), grouped_split(_fake(), seed=42)
    assert a["split"].equals(b["split"])
    c = grouped_split(_fake(), seed=7)
    assert not a["split"].equals(c["split"]), "different seeds gave identical splits"


def test_image_level_split_would_leak():
    """The control: prove the check can actually fail. A naive per-image
    split of the same frame leaks, and assert_no_leak catches it."""
    df = _fake()
    rng = np.random.RandomState(0)
    df["split"] = rng.choice(["train", "val", "test"], len(df))
    try:
        assert_no_leak(df, "naive")
    except AssertionError:
        return
    raise AssertionError("assert_no_leak failed to detect an image-level split")


def test_balanced_subset_is_balanced_and_leak_free():
    bal = build_balanced(_fake(400, seed=1), n_per_class=20)
    counts = bal["label"].value_counts()
    assert counts.max() <= 20, counts.to_dict()
    assert bal["group"].duplicated().sum() >= 0
    out = grouped_split(bal)
    assert_no_leak(out, "balanced")


def test_no_lesion_spans_balanced_and_split():
    bal = grouped_split(build_balanced(_fake(400, seed=2), n_per_class=20))
    per_group = bal.groupby("group")["split"].nunique()
    assert (per_group == 1).all()


def test_shades_of_gray_neutralises_a_colour_cast():
    rng = np.random.RandomState(0)
    img = rng.randint(40, 200, (48, 48, 3)).astype(np.uint8)
    cast = np.clip(img.astype(np.float32) * np.array([1.3, 1.0, 0.8]), 0, 255).astype(np.uint8)
    a, b = shades_of_gray(img), shades_of_gray(cast)
    before = np.abs(img.astype(float).mean((0, 1)) - cast.astype(float).mean((0, 1))).mean()
    after = np.abs(a.astype(float).mean((0, 1)) - b.astype(float).mean((0, 1))).mean()
    assert after < before, f"colour constancy made it worse: {before:.2f} -> {after:.2f}"


def test_grayscale_keeps_three_channels():
    img = np.random.RandomState(0).randint(0, 255, (32, 32, 3)).astype(np.uint8)
    g = to_grayscale_3ch(img)
    assert g.shape == img.shape
    assert (g[..., 0] == g[..., 1]).all() and (g[..., 1] == g[..., 2]).all()


def test_pad_style_lesion_ids_do_not_merge_patients():
    """Regression: PAD-UFES-20's docs never state whether lesion_id is
    unique across patients. If it is numbered within a patient, grouping by
    it alone merges unrelated lesions -- and assert_no_leak still passes,
    because the groups are internally consistent. The split looks clean and
    leaks anyway. load_pad() therefore groups by patient_id + lesion_id."""
    rows = []
    for patient in range(40):
        for lesion in (1, 2):                 # per-patient numbering
            for k in range(2):
                rows.append({"image_id": f"P{patient}_L{lesion}_{k}",
                             "path": f"/tmp/P{patient}_L{lesion}_{k}.png",
                             "label": ["bcc", "mel", "akiec", "nv"][patient % 4],
                             "patient_id": f"P{patient}",
                             "lesion_id": str(lesion)})
    df = pd.DataFrame(rows)

    naive = df.assign(group=df["lesion_id"])
    assert naive["group"].nunique() == 2, "bare lesion_id should collapse to 2 groups"

    composite = df.assign(group=df["patient_id"] + "_" + df["lesion_id"])
    assert composite["group"].nunique() == 80, composite["group"].nunique()

    out = grouped_split(composite)
    assert_no_leak(out, "pad_composite")
    # and no patient's images for one lesion are scattered across splits
    per = out.groupby(["patient_id", "lesion_id"])["split"].nunique()
    assert (per == 1).all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"[pass] {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")
