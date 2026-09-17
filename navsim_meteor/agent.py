"""NAVSIM AbstractAgent implementation backed by METEOR ONNX."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory

from navsim_meteor.geometry import build_meteor_inputs, meteor_ego_to_navsim_poses


class MeteorONNXAgent(AbstractAgent):
    """Run the released single-frame, camera-only METEOR planner in NAVSIM."""

    requires_scene = False

    def __init__(
        self,
        model_path: str,
        providers: Optional[List[str]] = None,
        trajectory_extension: str = "linear",
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ) -> None:
        super().__init__(trajectory_sampling)
        self._model_path = str(model_path)
        self._providers = providers
        self._trajectory_extension = trajectory_extension
        self._session = None
        self._input_names = set()
        self._ego_output_name = "ego"

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        import onnxruntime as ort

        model_path = Path(self._model_path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"METEOR ONNX model not found: {model_path}")

        available = ort.get_available_providers()
        providers = self._providers or [
            name for name in ("CUDAExecutionProvider", "CPUExecutionProvider") if name in available
        ]
        if not providers:
            raise RuntimeError(f"No usable ONNX Runtime provider; available={available}")
        missing = [name for name in providers if name not in available]
        if missing:
            raise RuntimeError(f"Requested ONNX providers unavailable: {missing}; available={available}")

        self._session = ort.InferenceSession(str(model_path), providers=providers)
        model_inputs = {item.name: item.type for item in self._session.get_inputs()}
        self._input_names = set(model_inputs)
        required = {"imgs", "K", "T_cam_ego", "v0"}
        missing_inputs = sorted(required - self._input_names)
        if missing_inputs:
            raise RuntimeError(f"METEOR graph is missing required inputs: {missing_inputs}")

        supported = required | {"intent", "lidar_bev", "lidar_flag"}
        unsupported = sorted(self._input_names - supported)
        if unsupported:
            raise RuntimeError(f"METEOR graph has unsupported inputs: {unsupported}")
        if model_inputs["imgs"] != "tensor(uint8)":
            raise RuntimeError(
                "This adapter requires the released uint8-input METEOR graph; "
                f"imgs has type {model_inputs['imgs']}"
            )
        if ("lidar_bev" in self._input_names) != ("lidar_flag" in self._input_names):
            raise RuntimeError("METEOR LiDAR graph must expose both lidar_bev and lidar_flag")

        output_names = {item.name for item in self._session.get_outputs()}
        if self._ego_output_name not in output_names:
            raise RuntimeError(f"METEOR graph has no 'ego' output; outputs={sorted(output_names)}")

    def get_sensor_config(self) -> SensorConfig:
        # NAVSIM exposes four history frames indexed 0..3. METEOR v157 is
        # single-frame, so only load the current frame to keep I/O bounded.
        current = [3]
        return SensorConfig(
            cam_f0=current,
            cam_l0=current,
            cam_l1=current,
            cam_l2=current,
            cam_r0=current,
            cam_r1=current,
            cam_r2=current,
            cam_b0=current,
            lidar_pc=False,
        )

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        if self._session is None:
            raise RuntimeError("MeteorONNXAgent.initialize() must be called before inference")
        if abs(float(self._trajectory_sampling.interval_length) - 0.5) > 1e-6:
            raise ValueError("METEOR trajectory conversion requires a 0.5-second sampling interval")

        feed = build_meteor_inputs(agent_input, self._input_names)
        if "lidar_bev" in self._input_names:
            # The integration is deliberately camera-only. A LiDAR-capable
            # graph remains valid because zero raster + flag=0 is the model's
            # documented no-LiDAR contract.
            feed["lidar_bev"] = np.zeros((1, 4, 400, 250), dtype=np.float32)
            feed["lidar_flag"] = np.zeros((1,), dtype=np.float32)

        ego = self._session.run([self._ego_output_name], feed)[0]
        poses = meteor_ego_to_navsim_poses(
            ego[0],
            output_count=self._trajectory_sampling.num_poses,
            extension=self._trajectory_extension,
        )
        return Trajectory(poses=poses, trajectory_sampling=self._trajectory_sampling)
