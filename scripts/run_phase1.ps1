# Phase 1 -- balanced pilot, three seeds. Windows PowerShell.
#   .\scripts\run_phase1.ps1
#
# The number that matters here is the STD of test macro-F1 across seeds:
# the noise floor. No effect claimed later may be smaller than it.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = $PWD

foreach ($s in 42, 43, 44) {
    python -m src.train --config configs/phase1_balanced.yaml --seed $s
    if ($LASTEXITCODE -ne 0) { throw "seed $s failed" }
}
python -m src.summarize --csv results/phase1_summary.csv

Write-Host ""
Write-Host "Read the std of test_macro_f1 across the three seeds above." -ForegroundColor Yellow
Write-Host "That is your noise floor. If it exceeds 0.05, raise epochs or seeds" -ForegroundColor Yellow
Write-Host "before starting Phase 2." -ForegroundColor Yellow
