param(
    [string]$Python = "python"
)

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

$PythonCommand = Get-Command $Python -ErrorAction Stop
Write-Host "Using Python: $($PythonCommand.Source)"
& $Python -c "import torch; print('torch', torch.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "Python interpreter cannot import torch. Re-run with a torch-enabled interpreter, for example: run_sweep.ps1 -Python <path-to-python.exe>"
}

Push-Location $Repo
try {
    foreach ($Script in $Scripts) {
        Write-Host "Running $($Script.FullName)"
        & $Python $Script.FullName
        if ($LASTEXITCODE -ne 0) {
            throw "Training script failed with exit code ${LASTEXITCODE}: $($Script.FullName)"
        }
    }
}
finally {
    Pop-Location
}
