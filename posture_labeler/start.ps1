param([int]$Port = 8765, [switch]$CheckEnvironment)
$ErrorActionPreference = 'Stop'
$labelerPython = Join-Path $env:USERPROFILE '.conda\envs\posture-mlp-v2\python.exe'
$labelerArguments = if ($CheckEnvironment) {
    @('-B', '-c', 'import sys, cv2, numpy; print(sys.executable)')
} else {
    @('-B', (Join-Path $PSScriptRoot 'server.py'), '--port', "$Port")
}
if (Test-Path -LiteralPath $labelerPython) {
    & $labelerPython @labelerArguments
} elseif (Get-Command conda -ErrorAction SilentlyContinue) {
    conda run --no-capture-output -n posture-mlp-v2 python @labelerArguments
} else {
    throw 'Create/activate posture-mlp-v2 using posture_mlp_v2/environment.yml, then run: python posture_labeler/server.py'
}
exit $LASTEXITCODE
