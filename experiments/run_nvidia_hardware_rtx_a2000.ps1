param(
    [string]$Python = "python",
    [string]$TrtExec = "",
    [int]$Seed = 42,
    [int]$Repeats = 5,
    [int]$DurationSeconds = 10,
    [int[]]$BatchSizes = @(1, 64)
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $RepoRoot

if (-not $TrtExec) {
    $candidate = Get-Command trtexec.exe -ErrorAction SilentlyContinue
    if ($candidate) {
        $TrtExec = $candidate.Source
    } elseif ($env:TENSORRT_ROOT) {
        $candidatePath = Join-Path $env:TENSORRT_ROOT "bin\trtexec.exe"
        if (Test-Path -LiteralPath $candidatePath) {
            $TrtExec = $candidatePath
        }
    }
}

if (-not $TrtExec -or -not (Test-Path -LiteralPath $TrtExec)) {
    throw "Không tìm thấy trtexec.exe. Truyền -TrtExec hoặc đặt TENSORRT_ROOT."
}

Write-Host "=== RTX A2000 TensorRT Level-3 benchmark ==="
& nvidia-smi
if ($LASTEXITCODE -ne 0) {
    throw "nvidia-smi không nhận GPU NVIDIA."
}

$arguments = @(
    "experiments/nvidia_sparse_benchmark.py",
    "--trtexec", $TrtExec,
    "--seed", "$Seed",
    "--repeats", "$Repeats",
    "--duration_s", "$DurationSeconds",
    "--warmup_ms", "2000",
    "--workspace_mb", "2048",
    "--batch_sizes"
) + ($BatchSizes | ForEach-Object { "$_" })

& $Python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Benchmark thất bại với exit code $LASTEXITCODE."
}
