"""Official no-target standing policy, using the supplied deployment contract.

Simulation adapter only. MuJoCo already reports robot-frame orientation, so the
physical IMU mounting correction must not be applied a second time here.
"""
from pathlib import Path
import hashlib
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UPPER_IDS = np.array([*range(6, 12), *range(18, 24), 24, 25, 26])
MODEL = ROOT / "policies/official/官方运动ONNX/policy_notarget/policy_base_v1.onnx"
MODEL_SHA256 = "ba2c5e761af49c80ffb3e165936e233288c8dd590d4fdfed9b986fd713276d10"


class StandingPolicy:
    frame_size = 165
    history_size = 10
    period = .02

    def __init__(self, default, lower, upper):
        sys.path.insert(0, str(ROOT / "vendor/python"))
        import onnxruntime as ort
        self.runtime_version = ort.__version__
        sys.path.insert(0, str(ROOT / "robot_bridge"))
        from mimic_bridge.trajectory import Trajectory

        model_bytes = MODEL.read_bytes()
        if hashlib.sha256(model_bytes).hexdigest() != MODEL_SHA256:
            raise ValueError("Official standing policy checksum mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_bytes, sess_options=options,
                                           providers=["CPUExecutionProvider"])
        inp, out = self.session.get_inputs()[0], self.session.get_outputs()[0]
        if inp.shape != [1, 1650] or out.shape != [1, 27] or inp.type != "tensor(float)":
            raise ValueError(f"Unexpected standing policy contract: {inp.shape}, {out.shape}")
        self.input_name, self.output_name = inp.name, out.name
        self.default = np.asarray(default, dtype=np.float32).copy()
        self.lower, self.upper = np.asarray(lower).copy(), np.asarray(upper).copy()
        self.lower[[4,16]] = -.52
        self.upper[[4,16]] = .61
        self.lower[[5,17]] = -.35
        self.upper[[5,17]] = .35
        velocity = np.zeros(27)
        velocity[UPPER_IDS] = 2.0
        velocity[24] = 0.0  # fixed-waist reference, as requested
        self.trajectory_config = dict(demo_lower=self.lower, demo_upper=self.upper,
                                      velocity_max=velocity, acceleration_max=np.full(27, 4.0))
        self.trajectory_type = Trajectory
        self.occupancy = np.ones(15, dtype=np.float32)
        self.reset(self.default)

    def reset(self, q):
        self.action = (np.asarray(q, dtype=np.float32) - self.default) / .5
        self.reference = self.trajectory_type(self.trajectory_config, self.default)
        self.history = None
        self.last_frame = None

    def make_frame(self, q, dq, omega, gravity, reference, reference_velocity):
        error = (np.asarray(reference) - q)[UPPER_IDS]
        frame = np.concatenate([
            np.asarray(omega) * .2, gravity, np.zeros(3),
            error, np.asarray(reference_velocity)[UPPER_IDS] * .2,
            error, error, self.occupancy,
            np.asarray(q) - self.default, np.asarray(dq) * .05, self.action,
        ]).astype(np.float32)
        if frame.shape != (165,) or not np.isfinite(frame).all():
            raise ValueError("Invalid standing observation")
        return frame

    def stack_frame(self, frame):
        if self.history is None:
            self.history = np.tile(frame, (10, 1))
        else:
            self.history[:-1] = self.history[1:]
            self.history[-1] = frame
        self.last_frame = frame.copy()
        return self.history.reshape(1, 1650)

    def step(self, q, dq, omega, gravity, requested):
        requested = np.asarray(requested).copy()
        requested[24] = self.default[24]
        reference, velocity = self.reference.step(requested, self.period)
        # Live camera has no trustworthy future trajectory. Hold the current
        # bounded reference in both future slots, matching the package's IK case.
        frame = self.make_frame(q, dq, omega, gravity, reference, velocity)
        obs = self.stack_frame(frame)
        result = self.session.run([self.output_name], {self.input_name: obs})[0][0]
        if result.shape != (27,) or not np.isfinite(result).all():
            raise ValueError("Invalid standing policy action")
        self.action = result.copy()
        target = np.clip(result * .5 + self.default, self.lower, self.upper)
        target[24] = self.default[24]
        return target.astype(np.float32)
