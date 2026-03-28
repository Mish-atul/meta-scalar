$ErrorActionPreference = 'Stop'

if (Get-Command python -ErrorAction SilentlyContinue) {
    python scripts/validate_pipeline.py
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    py -3 scripts/validate_pipeline.py
} else {
    Write-Error 'Python executable not found on PATH.'
}
