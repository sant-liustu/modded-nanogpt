$ErrorActionPreference = "Stop"

$script = Join-Path $PSScriptRoot "generated_train_scripts\train_small_batch_adamw_B0128_devB064_lr0p0036_wd0_seed00.py"
python $script
