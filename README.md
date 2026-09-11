# Cross-Domain Generalization in Skin Lesion Classification

How much does a classifier trained on dermoscopic images (HAM10000) degrade when
evaluated on smartphone clinical images (PAD-UFES-20), which factors cause the
drop, and how much target-domain data is needed to recover it?

---

## Ground rules

These are enforced by the code, not by discipline.

1. **Never overwrite results.** Every run writes to `results/<run_id>/`, where
   `run_id` is a timestamp + model + seed + config hash. `src/train.py` never
   writes to an existing directory.
2. **Every number traces to a `run_id`.** Each run directory holds
   `config.yaml`, `environment.json`, `metrics.json`, `test_predictions.npz`
   and `best.pt`. `python -m src.summarize` builds the table from those files
   alone.
3. **Splits are files, not functions.** `python -m src.data.build_splits`
   is run once; the CSVs in `data/splits/` are committed and never regenerated
   to fix a result.
4. **Group by lesion, not by image.** HAM10000 and PAD-UFES-20 both contain
   several images of the same lesion. Splitting by image leaks near-duplicates
   into the test set and inflates macro-F1 by several points — the most common
   error in papers using these datasets. Every split goes through
   `assert_no_leak()`, and `tests/test_splits.py` includes a control proving
   that check can actually fail.
5. **Report macro-F1 and per-class recall.** Accuracy is meaningless when one
   class is 67% of the data.
6. **Three seeds minimum** for any number that appears in the thesis.

---

## Setup

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
export PYTHONPATH=$PWD          # every entrypoint is run as `python -m src.…`
```

## Data

| Dataset | Canonical source | Notes |
|---|---|---|
| HAM10000 | Kaggle `kmader/skin-cancer-mnist-ham10000` (orig. Harvard Dataverse `10.7910/DVN/DBW86T`) | 10015 images, 7 classes |
| PAD-UFES-20 | **Mendeley Data `10.17632/zr7vgbcyr2.1`** | 2298 images, 1641 lesions, 1373 patients, 6 classes, `.png` |

```bash
python -m src.data.download --pad-source mendeley    # default; canonical
python -m src.data.download --pad-source kaggle      # mirror, schema unverified
```

`download.py` asserts HAM10000's row count and exact class distribution, and
asserts PAD-UFES-20's required column names (`patient_id`, `lesion_id`,
`img_id`, `diagnostic`, `age`, `region`, `biopsed`). A count mismatch on the
PAD mirror warns rather than raises, because a mirror can legitimately be a
subset — but report the distribution you actually got.

**If a mirror's column names differ, adapt `load_pad()` in
`src/data/build_splits.py` and record the change.** Do not rename columns
elsewhere in the pipeline.

### Label mapping

The shared label space is four classes. Every decision is argued in
`src/data/build_splits.py`; the short version:

| PAD-UFES-20 | HAM10000 | |
|---|---|---|
| `BCC` | `bcc` | same entity |
| `MEL` | `mel` | same entity |
| `NEV` | `nv` | same entity |
| `ACK` | `akiec` | HAM's `akiec` is broader (includes intraepithelial carcinoma) |
| `SEK` | `bkl` | **optional** 5th class, `--include-sek` |
| `SCC` | — | **excluded**: invasive SCC is not HAM's in-situ `akiec` |

`--include-sek` writes `*_shared5.csv` instead, for the committee's
sensitivity check.

---

## Phases

```bash
python -m src.data.build_splits        # once; writes data/splits/*.csv

bash scripts/run_phase1.sh             # balanced pilot, 3 seeds
bash scripts/run_phase2.sh             # full HAM10000, 12 runs
bash scripts/run_phase3.sh             # cross-domain
```

**Phase 1 (balanced pilot)** exists to surface bugs while runs take two minutes
instead of two hours. Do not tune for accuracy. Its deliverable is the
**standard deviation of test macro-F1 across the three seeds** — the noise
floor. If it exceeds 0.05, raise epochs or seeds before continuing; no effect
claimed later may be smaller than it.

**Phase 2 (full HAM10000)** is a 2 × 2 × 3 grid over model, imbalance
strategy and seed. Expect `df`, `vasc` and `akiec` to be visibly worse. That
asymmetry is a finding — write it down.

**Phase 3 (cross-domain)** is the contribution.

```bash
python -m src.cross_domain --run_id <RUN_ID> --mode zero_shot   # headline number
python -m src.cross_domain --run_id <RUN_ID> --mode ablation    # colour, eval-side
python -m src.cross_domain --run_id <RUN_ID> --mode curve       # recovery curve
```

> **Confound warning.** `--mode ablation` applies the colour transform to the
> *target at evaluation time only*, against a model trained on raw images. The
> drop it shows mixes the domain effect with a train/test preprocessing
> mismatch. The honest ablation trains a matched source model per colour mode
> (`--color_mode shades_of_gray|grayscale`), which is what `scripts/run_phase3.sh`
> step 2 does. Quote the matched version.

Hypothesis to state **before** running: if grayscale *shrinks* the gap, colour
dominates and shades-of-gray should recover part of it. If grayscale leaves the
gap unchanged, the difference is structural — dermoscopy resolves subsurface
patterns a smartphone physically cannot capture. Either outcome is publishable.

**Phase 4 (error analysis)**

```bash
python -m src.analysis.error_analysis --run_id <RUN_ID> --n 100
# fill failure_category by hand, then:
python -m src.analysis.error_analysis --run_id <RUN_ID> --summarize <csv>
python -m src.analysis.figures --run_id <RUN_ID>
```

Label all 100 by hand — roughly two hours, and it is the qualitative layer that
separates a thesis from a benchmark table. Do not automate that column.

---

## Verifying the pipeline without the datasets

```bash
python tests/test_splits.py      # unit tests, no pytest needed
bash tests/smoke_test.sh         # full pipeline on synthetic fixtures, CPU-only
```

`tests/make_synthetic.py` generates fake HAM10000 and PAD-UFES-20 trees with the
real schema, multi-image lesions and a deliberate colour-cast domain gap. The
smoke test runs splits → training → zero-shot → ablation → recovery curve →
error export → figures → summary. It checks that the pipeline is *wired*, not
that any result is meaningful; the fixture numbers mean nothing.

---

## Layout

```
configs/          one YAML per phase (+ smoke configs)
data/raw/         downloaded, never modified, never committed
data/splits/      committed CSVs — the source of truth for every run
src/data/         download.py, build_splits.py, datasets.py
src/              preprocessing.py, models.py, train.py, cross_domain.py, summarize.py
src/analysis/     error_analysis.py, figures.py
scripts/          run_phase{1,2,3}.sh
tests/            unit tests + synthetic-fixture smoke test
results/<run_id>/ config.yaml, environment.json, metrics.json, best.pt, predictions
```

## Risk register

| Risk | Mitigation |
|---|---|
| PAD-UFES-20 mirror unavailable or different schema | `--pad-source mendeley`; `verify_pad()` asserts columns and reports the distribution |
| Label mapping disputed by committee | every decision documented in `build_splits.py`; `--include-sek` runs the sensitivity check |
| Zero-shot gap is small | still a result; emphasis shifts to the ablations |
| Seed variance exceeds the effect | Phase 1 reports the noise floor first; raise to 5 seeds and report CIs |
| VRAM insufficient at 224px | `--batch_size 24`; do not trade resolution for batch, resolution is not the interesting variable |
| Disk failure | push daily; `results/**/*.pt` and `*.npz` are gitignored, `metrics.json` is not |
