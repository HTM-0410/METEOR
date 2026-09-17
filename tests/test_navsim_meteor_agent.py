import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

from navsim_meteor.geometry import METEOR_CAMERA_ORDER


class FakeTrajectorySampling:
    def __init__(self, time_horizon=4, interval_length=0.5):
        self.time_horizon = time_horizon
        self.interval_length = interval_length
        self.num_poses = int(round(time_horizon / interval_length))


class FakeAbstractAgent:
    def __init__(self, trajectory_sampling, requires_scene=False):
        self.requires_scene = requires_scene
        self._trajectory_sampling = trajectory_sampling


class FakeSensorConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeTrajectory:
    def __init__(self, poses, trajectory_sampling):
        self.poses = poses
        self.trajectory_sampling = trajectory_sampling


class FakeValueInfo:
    def __init__(self, name, value_type="tensor(float)"):
        self.name = name
        self.type = value_type


class FakeInferenceSession:
    last_feed = None

    def __init__(self, model_path, providers):
        self.model_path = model_path
        self.providers = providers

    def get_inputs(self):
        return [
            FakeValueInfo("imgs", "tensor(uint8)"),
            FakeValueInfo("K"),
            FakeValueInfo("T_cam_ego"),
            FakeValueInfo("v0"),
        ]

    def get_outputs(self):
        return [FakeValueInfo("ego")]

    def run(self, output_names, feed):
        FakeInferenceSession.last_feed = feed
        ego = np.zeros((1, 42), dtype=np.float32)
        ego[0, :12] = np.array([[index, 0] for index in range(1, 7)], dtype=np.float32).reshape(-1)
        ego[0, 36] = 5
        return [ego]


def _install_navsim_stubs(monkeypatch):
    modules = {}
    for name in (
        "nuplan",
        "nuplan.planning",
        "nuplan.planning.simulation",
        "nuplan.planning.simulation.trajectory",
        "nuplan.planning.simulation.trajectory.trajectory_sampling",
        "navsim",
        "navsim.agents",
        "navsim.agents.abstract_agent",
        "navsim.common",
        "navsim.common.dataclasses",
    ):
        modules[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, modules[name])
    modules["nuplan.planning.simulation.trajectory.trajectory_sampling"].TrajectorySampling = FakeTrajectorySampling
    modules["navsim.agents.abstract_agent"].AbstractAgent = FakeAbstractAgent
    modules["navsim.common.dataclasses"].AgentInput = object
    modules["navsim.common.dataclasses"].SensorConfig = FakeSensorConfig
    modules["navsim.common.dataclasses"].Trajectory = FakeTrajectory

    ort = ModuleType("onnxruntime")
    ort.get_available_providers = lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ort.InferenceSession = FakeInferenceSession
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)


def test_agent_initializes_preprocesses_and_returns_navsim_trajectory(monkeypatch, tmp_path: Path):
    _install_navsim_stubs(monkeypatch)
    sys.modules.pop("navsim_meteor.agent", None)
    agent_module = importlib.import_module("navsim_meteor.agent")

    model_path = tmp_path / "meteor.onnx"
    model_path.write_bytes(b"fake")
    agent = agent_module.MeteorONNXAgent(str(model_path), trajectory_sampling=FakeTrajectorySampling())
    agent.initialize()

    camera = SimpleNamespace(
        image=np.zeros((900, 1600, 3), dtype=np.uint8),
        intrinsics=np.eye(3, dtype=np.float32),
        sensor2lidar_rotation=np.eye(3, dtype=np.float32),
        sensor2lidar_translation=np.zeros(3, dtype=np.float32),
    )
    cameras = SimpleNamespace(**{navsim_name: camera for _, navsim_name in METEOR_CAMERA_ORDER})
    status = SimpleNamespace(
        ego_velocity=np.array([2.0, 0.0], dtype=np.float32),
        driving_command=np.array([0, 1, 0, 0], dtype=np.float32),
    )
    trajectory = agent.compute_trajectory(SimpleNamespace(cameras=[cameras], ego_statuses=[status]))

    assert trajectory.poses.shape == (8, 3)
    np.testing.assert_allclose(trajectory.poses[:, 0], np.arange(1, 9))
    assert FakeInferenceSession.last_feed["imgs"].shape == (1, 8, 3, 432, 768)
    sensor_config = agent.get_sensor_config()
    assert sensor_config.cam_f0 == [3]
    assert sensor_config.lidar_pc is False
