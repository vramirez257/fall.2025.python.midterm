# run_all.ps1  (SAFE: batches outside outputs)
$py     = "python"
$script = ".\midterm_pipeline.py"
$data   = ".\laptop_pc_sales_dataset (1).xlsx"

# Keep per-run archives OUTSIDE outputs/
$stamp   = Get-Date -Format "yyyyMMdd-HHmmss"
$archive = Join-Path (Get-Location) ("batches\batch-" + $stamp)
New-Item -ItemType Directory -Path $archive -Force | Out-Null

function RunStage([string]$stageName, [string[]]$argsArray) {
  Write-Host "`n=== Running $stageName ==="
  & $py $script @argsArray
  if ($LASTEXITCODE -ne 0) { throw "Stage '$stageName' failed with exit code $LASTEXITCODE." }

  # Make destination folder for this stage's artifacts
  $dest = Join-Path $archive $stageName
  New-Item -ItemType Directory -Path $dest -Force | Out-Null

  # Copy whatever the script wrote this run (if any)
  if (Test-Path ".\outputs") {
    Copy-Item ".\outputs\*" $dest -Recurse -Force -ErrorAction SilentlyContinue
    # Clean ONLY the files that the script writes, not the archive location
    Remove-Item ".\outputs\*" -Recurse -Force -ErrorAction SilentlyContinue
  }
}

# A) Revenue regression
RunStage "A_Revenue" @("--data", $data, "--excel", "--target", "TotalAmount")

# B) Inventory classification
RunStage "B_Inventory" @("--data", $data, "--excel", "--target", "InventoryAction", "--threshold", "0.35")

# C) HighRating classification + What-If
RunStage "C_HighRating" @("--data", $data, "--excel", "--target", "HighRating", "--threshold", "0.40")

# D) Forecast with seasonality
RunStage "D_ForecastSeasonality" @("--data", $data, "--excel", "--target", "TotalAmount", "--add-seasonality")

Write-Host "`nAll done. See per-step outputs in: $archive"
Invoke-Item $archive  # open in Explorer
