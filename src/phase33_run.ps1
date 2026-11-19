# Run only Phase 3.3 artifacts with the frozen project interpreter and local caches.
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Project .venv Python is missing.' }
$env:PIP_CACHE_DIR = Join-Path $projectRoot '.cache\pip'
$env:HF_HOME = Join-Path $projectRoot '.cache\huggingface'
$env:HF_HUB_CACHE = Join-Path $projectRoot '.cache\huggingface\hub'
$env:TORCH_HOME = Join-Path $projectRoot '.cache\torch'
$env:TEMP = Join-Path $projectRoot '.cache\tmp'
$env:TMP = Join-Path $projectRoot '.cache\tmp'
Push-Location $projectRoot
try {
    foreach ($scale in @('medium', 'large')) {
        foreach ($method in @('fixed', 'sentence')) {
            $runId = "phase33_${scale}_${method}"
            $resultDir = Join-Path $projectRoot "experiments\results\phase33\$runId"
            $stats = Join-Path $resultDir 'chunking_stats.json'
            if (-not (Test-Path -LiteralPath $stats)) {
                & $python 'src\phase33_chunk.py' --scale $scale --method $method --mode build
                if ($LASTEXITCODE -ne 0) { throw "$runId chunk build failed" }
            }
            $repeats = Join-Path $resultDir 'chunking_repeats.json'
            while ((-not (Test-Path -LiteralPath $repeats)) -or
                   ((Get-Content -LiteralPath $repeats -Raw | ConvertFrom-Json).runs.Count -lt 3)) {
                & $python 'src\phase33_chunk.py' --scale $scale --method $method --mode benchmark
                if ($LASTEXITCODE -ne 0) { throw "$runId chunk benchmark failed" }
            }
            if (-not (Test-Path -LiteralPath (Join-Path $resultDir 'embedding_metrics.json'))) {
                & $python 'src\phase33_embed.py' --scale $scale --method $method
                if ($LASTEXITCODE -ne 0) { throw "$runId embedding failed" }
            }
            if (-not (Test-Path -LiteralPath (Join-Path $resultDir 'index_metrics.json'))) {
                & $python 'src\faiss_builder.py' --run-id $runId
                if ($LASTEXITCODE -ne 0) { throw "$runId index failed" }
            }
            if (-not (Test-Path -LiteralPath (Join-Path $resultDir 'metrics.json'))) {
                & $python 'src\phase31_evaluate.py' --run-id $runId
                if ($LASTEXITCODE -ne 0) { throw "$runId evaluation failed" }
            }
            Write-Output "$runId complete"
        }
    }
}
finally { Pop-Location }
