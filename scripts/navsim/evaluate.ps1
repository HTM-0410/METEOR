param(
    [Parameter(Mandatory = $true)] [string] $NavsimRoot,
    [Parameter(Mandatory = $true)] [string] $ModelPath,
    [Parameter(Mandatory = $true)] [string] $DataRoot,
    [Parameter(Mandatory = $true)] [string] $ExpRoot,
    [string] $Split = "navhard_two_stage",
    [string] $MetricCachePath = "",
    [string] $PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$meteorRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$navsimRootResolved = (Resolve-Path -LiteralPath $NavsimRoot).Path
$modelResolved = (Resolve-Path -LiteralPath $ModelPath).Path
$dataResolved = (Resolve-Path -LiteralPath $DataRoot).Path
$expResolved = if (Test-Path -LiteralPath $ExpRoot) {
    (Resolve-Path -LiteralPath $ExpRoot).Path
} else {
    (New-Item -ItemType Directory -Path $ExpRoot).FullName
}
if (-not $MetricCachePath) {
    $MetricCachePath = Join-Path $expResolved "metric_cache"
}
if (-not (Test-Path -LiteralPath $MetricCachePath)) {
    throw "Metric cache not found: $MetricCachePath. Build it with NAVSIM scripts/evaluation/run_metric_caching.sh before scoring."
}

$runScript = Join-Path $navsimRootResolved "navsim\planning\script\run_pdm_score.py"
if (-not (Test-Path -LiteralPath $runScript)) {
    throw "NAVSIM run_pdm_score.py not found under: $navsimRootResolved"
}

$env:NAVSIM_DEVKIT_ROOT = $navsimRootResolved
$env:OPENSCENE_DATA_ROOT = $dataResolved
$env:NAVSIM_EXP_ROOT = $expResolved
$env:METEOR_ONNX_PATH = $modelResolved
$env:NUPLAN_MAP_VERSION = "nuplan-maps-v1.0"
$env:NUPLAN_MAPS_ROOT = Join-Path $dataResolved "maps"
$env:PYTHONPATH = "$meteorRoot;$navsimRootResolved;$env:PYTHONPATH"

$arguments = @(
    $runScript,
    "agent=constant_velocity_agent",
    "agent._target_=navsim_meteor.agent.MeteorONNXAgent",
    "+agent.model_path=$modelResolved",
    "+agent.providers=null",
    "+agent.trajectory_extension=linear",
    "train_test_split=$Split",
    "experiment_name=meteor_$Split",
    "metric_cache_path=$MetricCachePath",
    "worker=sequential",
    "max_number_of_workers=1"
)

if ($Split -like "*_two_stage") {
    $arguments += "synthetic_sensor_path=$(Join-Path $dataResolved "$Split\sensor_blobs")"
    $arguments += "synthetic_scenes_path=$(Join-Path $dataResolved "$Split\synthetic_scene_pickles")"
}

Write-Host "Running METEOR on NAVSIM split '$Split' (sequential worker)..."
& $PythonExe @arguments
if ($LASTEXITCODE -ne 0) {
    throw "NAVSIM evaluation failed with exit code $LASTEXITCODE"
}
