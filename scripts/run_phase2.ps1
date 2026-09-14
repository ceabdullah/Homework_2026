# Phase 2 -- full HAM10000. 2 models x 2 imbalance strategies x 3 seeds = 12
# runs, roughly 25-40 min each on a 3080 Ti at 224px (so budget ~6 hours).
#   .\scripts\run_phase2.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = $PWD

foreach ($m in "resnet50", "convnext_tiny") {
    foreach ($imb in "none", "class_weighted") {
        foreach ($s in 42, 43, 44) {
            python -m src.train --config configs/phase2_full.yaml `
                --model $m --imbalance $imb --seed $s
            if ($LASTEXITCODE -ne 0) { throw "run failed: $m / $imb / seed $s" }
        }
    }
}
python -m src.summarize --csv results/phase2_summary.csv
