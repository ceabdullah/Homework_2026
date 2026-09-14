# Phase 3 -- cross-domain. The contribution; everything before it is setup.
#   .\scripts\run_phase3.ps1
#
# Note on the colour ablation. `cross_domain.py --mode ablation` transforms the
# TARGET at evaluation time only, against a model trained on raw images. That is
# cheap but confounded: the drop it shows mixes the domain effect with a
# train/test preprocessing mismatch. The honest version trains a matched source
# model per colour mode, which is what step 2 below does. Quote that one.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = $PWD

# 1. source baselines in the shared label space
foreach ($m in "resnet50", "convnext_tiny") {
    foreach ($s in 42, 43, 44) {
        python -m src.train --config configs/phase3_crossdomain.yaml --model $m --seed $s
        if ($LASTEXITCODE -ne 0) { throw "source run failed: $m / seed $s" }
    }
}

# 2. matched colour-mode source models (the unconfounded ablation)
foreach ($cm in "shades_of_gray", "grayscale") {
    foreach ($s in 42, 43, 44) {
        python -m src.train --config configs/phase3_crossdomain.yaml `
            --model resnet50 --color_mode $cm --seed $s
        if ($LASTEXITCODE -ne 0) { throw "ablation run failed: $cm / seed $s" }
    }
}

# 3. zero-shot every shared-label-space run onto PAD
python -m src.zero_shot_all

python -m src.summarize --csv results/phase3_summary.csv

Write-Host ""
Write-Host "Now pick the best source run_id from the table above and run:"
Write-Host "  python -m src.cross_domain --run_id <RUN_ID> --mode curve"
Write-Host "  python -m src.analysis.error_analysis --run_id <RUN_ID>"
Write-Host "  python -m src.analysis.figures --run_id <RUN_ID>"
