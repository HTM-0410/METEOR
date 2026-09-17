#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 NAVSIM_ROOT METEOR_ONNX DATA_ROOT EXP_ROOT [SPLIT] [METRIC_CACHE]" >&2
  exit 2
fi

NAVSIM_ROOT=$(cd "$1" && pwd)
METEOR_ONNX=$(realpath "$2")
OPENSCENE_DATA_ROOT=$(cd "$3" && pwd)
NAVSIM_EXP_ROOT=$(mkdir -p "$4" && cd "$4" && pwd)
SPLIT=${5:-navhard_two_stage}
METRIC_CACHE=${6:-$NAVSIM_EXP_ROOT/metric_cache}
METEOR_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

[[ -f "$NAVSIM_ROOT/navsim/planning/script/run_pdm_score.py" ]] || {
  echo "NAVSIM run_pdm_score.py not found under $NAVSIM_ROOT" >&2; exit 2;
}
[[ -f "$METEOR_ONNX" ]] || { echo "METEOR ONNX not found: $METEOR_ONNX" >&2; exit 2; }
[[ -d "$METRIC_CACHE" ]] || {
  echo "Metric cache not found: $METRIC_CACHE. Run NAVSIM metric caching first." >&2; exit 2;
}

export NAVSIM_ROOT METEOR_ONNX OPENSCENE_DATA_ROOT NAVSIM_EXP_ROOT
export NAVSIM_DEVKIT_ROOT="$NAVSIM_ROOT"
export METEOR_ONNX_PATH="$METEOR_ONNX"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export PYTHONPATH="$METEOR_ROOT:$NAVSIM_ROOT:${PYTHONPATH:-}"

extra=()
if [[ "$SPLIT" == *_two_stage ]]; then
  extra+=("synthetic_sensor_path=$OPENSCENE_DATA_ROOT/$SPLIT/sensor_blobs")
  extra+=("synthetic_scenes_path=$OPENSCENE_DATA_ROOT/$SPLIT/synthetic_scene_pickles")
fi

python "$NAVSIM_ROOT/navsim/planning/script/run_pdm_score.py" \
  agent=constant_velocity_agent \
  agent._target_=navsim_meteor.agent.MeteorONNXAgent \
  "+agent.model_path=$METEOR_ONNX" \
  +agent.providers=null \
  +agent.trajectory_extension=linear \
  "train_test_split=$SPLIT" \
  "experiment_name=meteor_$SPLIT" \
  "metric_cache_path=$METRIC_CACHE" \
  worker=sequential \
  max_number_of_workers=1 \
  "${extra[@]}"
