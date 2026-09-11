# Lab notebook

Append-only. Every entry: date, what was run, what happened, what it means.
The "Limitations" chapter is written from this file, so record the failures
in more detail than the successes.

---

## 2026-09-11 — Phase 0 scaffold, pipeline verified on fixtures

**Environment:** Linux container, CPU only, no GPU, no dataset access.
Egress policy blocks `kaggle.com`, `data.mendeley.com`, `arxiv.org`,
`huggingface.co` and `download.pytorch.org`. PyPI is reachable.

**Consequences, all of which must be redone on the workstation:**

- `pip install torch==2.5.1 --index-url download.pytorch.org/whl/cpu` fails
  (host blocked). The container already carried **torch 2.14.0+cu130**, and
  that is what the smoke test ran on — *not* the pinned 2.5.1. The pin in
  `requirements.txt` is still the right one for the RTX 3080 Ti; it is simply
  unverified here.
- `timm.create_model(..., pretrained=True)` fetches from the HF hub, which is
  blocked. Every run here used `pretrained: false` (random init). **No number
  produced in this container is comparable to a real run.**
- Neither dataset could be downloaded. `verify_ham` / `verify_pad` have
  therefore never executed against real files. Run
  `python -m src.data.download` first thing on the workstation and paste its
  output into this file — that is the Phase 0 acceptance criterion.
- The PAD-UFES-20 schema encoded in `load_pad()` (26 columns; `patient_id`,
  `lesion_id`, `img_id`, `diagnostic`, `age`, `region`, `biopsed`; `.png`
  images; 2298 images / 1641 lesions / 1373 patients; BCC 845, ACK 730,
  NEV 244, SEK 235, SCC 192, MEL 52) comes from the published dataset
  description, **not** from inspecting the files. `verify_pad()` checks it at
  download time. If it fires, adapt `load_pad()` and record the deviation here.

**What was verified:** `tests/test_splits.py` (7/7) and `tests/smoke_test.sh`
end to end on synthetic fixtures — splits, leakage assertions, training loop,
checkpointing, zero-shot transfer, colour ablation, few-shot curve, error
export, all five figures, summary table.

**Bugs found and fixed while building (each would have cost real time later):**

1. `albumentations` was missing from `requirements.txt` although `train.py`
   imports it. Fresh env would have died on the first run.
2. `evaluate()` computed macro-F1 without `labels=`. On the cross-domain
   target some source classes are absent, so sklearn averaged over the
   *present* classes only — making in-domain and cross-domain macro-F1
   silently incomparable, i.e. corrupting the headline number. Now pinned to
   the full label space, with `macro_f1_present_only` reported alongside.
3. `torch.amp.GradScaler("cuda")` / `autocast("cuda")` were unconditional and
   crash on CPU. Now gated on device.
4. Phase 3 built the source split by *filtering* the 7-class split, which
   leaves uneven train/val/test proportions after the class filter. Now
   re-split after filtering.
5. Fine-tuning wrote fixed paths `/tmp/ft_{seed}.csv`, which collide between
   concurrent runs. Now a per-run `tempfile.TemporaryDirectory`.
6. `torch.load` without `weights_only=True` (warns now, will break later).
7. `build_balanced` could silently return fewer images than requested; it now
   prints realized counts.
8. `--mode ablation` is confounded — it transforms the target at eval time
   against a raw-trained model, mixing the domain effect with a train/test
   preprocessing mismatch. Kept (it is cheap), but the matched-training
   version was added to `scripts/run_phase3.sh` and the README says to quote
   that one.

**Not done:** Phases 1–4 proper. They need the datasets and a GPU.

---

## <date> — Phase 0 on the workstation

- [ ] `python -m src.data.download` output pasted here, PAD source named
- [ ] HAM class distribution matches nv 6705 / mel 1113 / bkl 1099 / bcc 514 /
      akiec 327 / vasc 142 / df 115
- [ ] PAD column names recorded verbatim
- [ ] `python -m src.data.build_splits` — all four `assert_no_leak` pass

## <date> — Phase 1 acceptance

- [ ] three seeds complete
- [ ] **test macro-F1 std across seeds = ______**  ← the noise floor, quote it
      before claiming any later effect
