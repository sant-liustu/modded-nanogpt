$ErrorActionPreference = "Stop"

$ExperimentDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Resolve-Path (Join-Path $ExperimentDir "..\..")
$ScriptDir = Join-Path $ExperimentDir "generated_train_scripts"

if (-not (Test-Path -LiteralPath $ScriptDir)) {
    throw "Missing generated script directory: $ScriptDir. Run generate_train_scripts.py first."
}

$Scripts = Get-ChildItem -LiteralPath $ScriptDir -Filter "train_B*_blockwd*_seed*.py" | Sort-Object Name
if ($Scripts.Count -eq 0) {
    throw "No generated training scripts found in $ScriptDir. Run generate_train_scripts.py first."
}

Push-Location $Repo
try {
    foreach ($Script in $Scripts) {
        Write-Host "Running $($Script.FullName)"
        python $Script.FullName
        if ($LASTEXITCODE -ne 0) {
            throw "Training script failed with exit code ${LASTEXITCODE}: $($Script.FullName)"
        }
    }
}
finally {
    Pop-Location
}
