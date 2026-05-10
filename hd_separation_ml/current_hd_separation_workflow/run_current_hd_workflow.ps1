$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$MlDir = Join-Path $RepoRoot "hd_separation_ml"
$Fig4Dir = Join-Path $RepoRoot "0-Figures\Figure4"
$Python = "C:\Users\yibinjiang\anaconda3\python.exe"

Write-Host "Step 1/4: all-28 classification feature screening"
Push-Location $MlDir
& $Python ".\bruteforce_classification_then_forward_regression_seed42_all28pool.py"

Write-Host "Step 2/4: RF/ExtraTrees-only forward addition"
& $Python ".\forward_regression_seed42_all28pool_rf_et_only.py"

Write-Host "Step 3/4: final 10-feature model, SHAP, and diagnostics"
& $Python ".\plot_10feature_seed42_rf_et_shap_importance.py"
Pop-Location

Write-Host "Step 4/4: BO overlay in top descriptor space"
Push-Location $Fig4Dir
& $Python ".\plot_top3_shap_descriptor_bo_overlay.py"
Pop-Location

Write-Host "Current H/D separation ML workflow completed."

