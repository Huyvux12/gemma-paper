$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    uv venv --python 3.13 .venv
    if ($LASTEXITCODE -ne 0) { throw 'uv venv failed' }
}
uv pip install --python .venv/Scripts/python.exe -r requirements_cpu.txt -r requirements_notebooks.txt
if ($LASTEXITCODE -ne 0) { throw 'CPU/notebook dependency install failed' }
uv pip install --python .venv/Scripts/python.exe torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'CPU torch install failed' }
Write-Output 'Ready: .venv/Scripts/python.exe -m jupyterlab notebooks'
