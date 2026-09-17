# NAVSIM evaluation

This adapter runs the released METEOR ONNX planner as a NAVSIM `AbstractAgent` and scores its
trajectory with NAVSIM's official PDM/EPDMS implementation. It is an **experimental zero-shot
cross-dataset evaluation**, not a claim that METEOR was trained or calibrated for OpenScene.

## Compatibility boundary

- Target devkit: NAVSIM v2.2 (the official 2025 challenge devkit). The adapter also follows the
  current `main` agent API as of commit `0a380a9063d7162ec93d0f51e9990ebac585f720`.
- METEOR input: all eight current NAVSIM RGB cameras, resized to 768x432 with matching intrinsic
  scaling. NAVSIM camera-to-LiDAR extrinsics are inverted into METEOR's ego-to-camera convention.
- Coordinates: both planners use x-forward, y-left, metres. OpenScene's merged-LiDAR frame is used
  as METEOR's local ego frame.
- Output: the highest-logit METEOR mode provides six poses at 0.5 s. NAVSIM requires a 4 s plan;
  the adapter linearly extrapolates the final learned 0.5 s displacement for the 3.5 s and 4.0 s
  poses. Set `agent.trajectory_extension=hold` for the conservative alternative.
- Route command: `meteor_v157c3Z.onnx` does not expose an `intent` input. The adapter maps NAVSIM's
  command when a future METEOR ONNX does expose it, but v157 is evaluated command-blind. This is
  expected to be a material limitation on turns and lane changes.
- LiDAR: evaluation is camera-only. A LiDAR-capable METEOR graph receives its documented zero
  raster and `lidar_flag=0` contract.

## 1. Install NAVSIM separately

NAVSIM pins Python 3.9, Torch 2.0.1 and NumPy 1.23.4. Keep it in a separate environment from any
newer METEOR training environment.

```bash
git clone --branch v2.2 https://github.com/autonomousvision/navsim.git ~/navsim_workspace/navsim
cd ~/navsim_workspace/navsim
conda env create --name navsim -f environment.yml
conda activate navsim
pip install -e .
pip install 'onnxruntime-gpu>=1.18,<2' huggingface_hub
hf download AutowareFoundation/meteor meteor_v157c3Z.onnx --local-dir ~/navsim_workspace/models
```

The plain `onnxruntime` package is sufficient for a CPU smoke run, but a complete NAVSIM split is
not practical on CPU. Confirm `CUDAExecutionProvider` is available:

```bash
python -c "import onnxruntime as o; print(o.get_available_providers())"
```

## 2. Download data and build the official cache

Follow NAVSIM's license and download instructions for maps, OpenScene logs/sensors and the desired
split. For a comparable local NAVSIM v2 score use `navhard_two_stage`; for local parity with the
warmup leaderboard use `warmup_two_stage`. Do not train on any evaluation split.

Set the standard variables and cache the same split that will be evaluated:

```bash
export NAVSIM_DEVKIT_ROOT=~/navsim_workspace/navsim
export OPENSCENE_DATA_ROOT=~/navsim_workspace/dataset
export NAVSIM_EXP_ROOT=~/navsim_workspace/exp
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=$OPENSCENE_DATA_ROOT/maps

cd $NAVSIM_DEVKIT_ROOT/scripts/evaluation
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
  train_test_split=navhard_two_stage \
  metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache
```

NAVSIM's `navhard_two_stage` download contains follow-up scene sensor blobs and scene pickles, but
its stage-one frames still use the OpenScene `test` logs and camera blobs. Both are required.

## 3. Run METEOR scoring

Linux/macOS:

```bash
./scripts/navsim/evaluate.sh \
  ~/navsim_workspace/navsim \
  ~/navsim_workspace/models/meteor_v157c3Z.onnx \
  ~/navsim_workspace/dataset \
  ~/navsim_workspace/exp \
  navhard_two_stage
```

PowerShell:

```powershell
.\scripts\navsim\evaluate.ps1 `
  -NavsimRoot C:\navsim_workspace\navsim `
  -ModelPath C:\navsim_workspace\models\meteor_v157c3Z.onnx `
  -DataRoot C:\navsim_workspace\dataset `
  -ExpRoot C:\navsim_workspace\exp `
  -Split navhard_two_stage
```

The wrapper deliberately uses NAVSIM's sequential worker. Parallel workers would each load the
large ONNX graph and can exhaust an 8 GB GPU. Results are written by NAVSIM under
`$NAVSIM_EXP_ROOT/meteor_<split>/<timestamp>/`.

The wrapper reuses NAVSIM's built-in constant-velocity Hydra schema and overrides its `_target_`
with `MeteorONNXAgent`; this avoids copying files into the NAVSIM checkout. The equivalent reusable
agent config is kept at `navsim_meteor/config/agent/meteor_agent.yaml`.

## 4. Interpret results honestly

Report the aggregate EPDMS/PDM score and every component in NAVSIM's output CSV, the exact NAVSIM
revision/split, METEOR model SHA-256, extension policy, valid/failed scenario count and ONNX
provider. Do not compare the score directly with METEOR's internal validation: camera rigs,
domains, horizon (3 s vs 4 s), route conditioning and metrics differ.

Run the adapter's geometry tests before a long evaluation:

```bash
pytest -q tests/test_navsim_meteor_geometry.py
```
