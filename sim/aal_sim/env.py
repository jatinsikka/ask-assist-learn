"""Machine-tending cell as a Gymnasium environment.

SOP: PICK raw part -> LOAD into fixture -> MACHINING (robot waits) -> UNLOAD finished part -> PLACE in bin -> DONE.

The current SOP step is derived from world state every control step, so a dropped part
automatically sends the robot back to the right step (e.g. LOAD -> PICK), which counts
as a regression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .ik import ARM_JOINTS, TopDownIK

ASSET = Path(__file__).parent / "assets" / "cell.xml"


class Step(IntEnum):
    PICK = 0
    LOAD = 1
    MACHINING = 2
    UNLOAD = 3
    PLACE = 4
    DONE = 5


STEP_NAMES = [s.name for s in Step]

# Geometry constants (metres), matching cell.xml.
PART_HALF_Z = 0.03
MACHINE_TOP = 0.10
SEAT_Z = MACHINE_TOP + PART_HALF_Z          # part centre when seated
GRASP_OFFSET = 0.015                         # tcp sits this far above the part centre
FINGER_TRAVEL = 0.04
BIN_INNER = 0.085
MACHINE_HALF = 0.1

RAW_RGBA = np.array([0.62, 0.55, 0.45, 1.0])
DONE_RGBA = np.array([0.05, 0.62, 0.58, 1.0])
LIGHT_IDLE = np.array([0.2, 0.8, 0.3, 1.0])
LIGHT_RUN = np.array([0.95, 0.65, 0.0, 1.0])


def wrap_quarter(a: float) -> float:
    """Wrap an angle to [-pi/4, pi/4): the part and fixture are square, so yaw is defined mod 90 deg."""
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


def wrap_pi(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def mat_yaw(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


@dataclass
class FaultConfig:
    """Per-episode fault probabilities. Each fault is sampled independently at reset."""
    ood_pose: float = 0.0       # raw part arrives far from the nominal tray pose, any yaw
    fixture_shift: float = 0.0  # fixture is offset up to 25 mm and rotated up to 17 deg
    slip: float = 0.0           # grip friction collapses briefly while carrying -> drop

    @classmethod
    def preset(cls, name: str) -> "FaultConfig":
        return {
            "none": cls(),
            "mild": cls(0.15, 0.15, 0.1),
            "hard": cls(0.35, 0.35, 0.25),
        }[name]


@dataclass
class EpisodeStats:
    regressions: int = 0
    drops: int = 0
    manual_fixes: int = 0
    max_step: int = 0
    faults: list = field(default_factory=list)


class MachineTendingEnv(gym.Env):
    """Top-down end-effector control of a 5-DOF arm tending a CNC machine.

    Action (5,) in [-1, 1]: tcp dx, dy, dz, dyaw, gripper (>0 closes).
    Observation (37,): see `_obs`. `info` carries the privileged state used by the scripted operator.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    control_dt = 0.05
    pos_scale = 0.015      # m per step at |a| = 1  (0.3 m/s)
    yaw_scale = 0.12       # rad per step at |a| = 1
    cycle_time = 3.0       # machining time, seconds
    max_time = 45.0
    obs_dim = 37
    workspace_lo = np.array([0.05, -0.6, 0.035])
    workspace_hi = np.array([0.70, 0.6, 0.45])
    home_pos = np.array([0.30, -0.05, 0.30])

    def __init__(self, faults: FaultConfig | str = "none", render_mode: str | None = None,
                 camera: str = "overview", width: int = 640, height: int = 480):
        self.model = mujoco.MjModel.from_xml_path(str(ASSET))
        self.data = mujoco.MjData(self.model)
        self.faults = FaultConfig.preset(faults) if isinstance(faults, str) else faults
        self.render_mode = render_mode
        self.camera = camera
        self.width, self.height = width, height
        self._renderer = None
        self._viewer = None

        m = self.model
        self.frame_skip = int(round(self.control_dt / m.opt.timestep))
        self.ik = TopDownIK(m)
        self.arm_qadr = self.ik.qadr
        self.finger_qadr = np.array([m.joint("finger_l").qposadr[0], m.joint("finger_r").qposadr[0]])
        self.part_qadr = m.joint("part_free").qposadr[0]
        self.part_dadr = m.joint("part_free").dofadr[0]
        self.part_body = m.body("part").id
        self.part_geom = m.geom("part").id
        self.finger_geoms = {m.geom("finger_l").id, m.geom("finger_r").id}
        self.fixture_body = m.body("fixture").id
        self.machine_body = m.body("machine").id
        self.bin_body = m.body("bin").id
        self.tcp_site = m.site("tcp").id
        self.light_geom = m.geom("status_light").id
        self.slip_geoms = [self.part_geom, *sorted(self.finger_geoms)]
        self.base_friction = m.geom_friction[self.slip_geoms].copy()
        self.nominal_fixture = m.body_pos[self.fixture_body].copy()
        self.nominal_machine = m.body_pos[self.machine_body].copy()

        self.action_space = spaces.Box(-1.0, 1.0, (5,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        rng = self.np_random
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)

        fc = options.get("faults", self.faults)
        fc = FaultConfig.preset(fc) if isinstance(fc, str) else fc
        active = [k for k in ("ood_pose", "fixture_shift", "slip") if rng.random() < getattr(fc, k)]
        for k in options.get("force_faults", []):
            if k not in active:
                active.append(k)
        self.stats = EpisodeStats(faults=active)

        # Fixture pose.
        if "fixture_shift" in active:
            off = rng.uniform(-0.025, 0.025, 2)
            fyaw = rng.uniform(-0.3, 0.3)
        else:
            off = rng.uniform(-0.003, 0.003, 2)
            fyaw = rng.uniform(-0.03, 0.03)
        m.body_pos[self.fixture_body] = self.nominal_fixture + np.array([off[0], off[1], 0.0])
        m.body_quat[self.fixture_body] = [np.cos(fyaw / 2), 0, 0, np.sin(fyaw / 2)]

        # Raw part pose.
        if "ood_pose" in active:
            xy = np.array([rng.uniform(0.30, 0.60), rng.uniform(-0.42, -0.08)])
            pyaw = rng.uniform(-np.pi, np.pi)
        else:
            xy = np.array([rng.uniform(0.40, 0.50), rng.uniform(-0.30, -0.20)])
            pyaw = rng.uniform(-0.35, 0.35)
        d.qpos[self.part_qadr:self.part_qadr + 3] = [xy[0], xy[1], PART_HALF_Z + 0.001]
        d.qpos[self.part_qadr + 3:self.part_qadr + 7] = [np.cos(pyaw / 2), 0, 0, np.sin(pyaw / 2)]
        m.geom_friction[self.slip_geoms] = self.base_friction
        m.geom_rgba[self.part_geom] = RAW_RGBA
        m.geom_rgba[self.light_geom] = LIGHT_IDLE

        # Arm at home, gripper open.
        home = rng.normal(0, 0.01, 3) + self.home_pos
        q = self.ik.solve(np.array([0.0, 0.3, -1.2, 0.9, 0.0]), home, 0.0, iters=60)
        d.qpos[self.arm_qadr] = q
        self.q_cmd = q.copy()
        d.ctrl[:5] = q
        d.ctrl[5:7] = 0.0
        mujoco.mj_forward(m, d)
        for _ in range(100):
            mujoco.mj_step(m, d)

        self.tcp_target = self.tcp_pos.copy()
        self.yaw_target = self.tcp_yaw
        self.grip_cmd = 0.0
        self.time = 0.0
        self.finished = False
        self.machining_started_at = None
        self.slip_at = None
        self.slip_until = None
        self.carry_started_at = None
        self.step_now = self._sop_step()
        self.prev_step = self.step_now
        return self._obs(), self._info()

    # ------------------------------------------------------------------ step
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        m, d = self.model, self.data

        tgt = self.tcp_target + a[:3] * self.pos_scale
        tgt = np.clip(tgt, self.workspace_lo, self.workspace_hi)
        tcp = self.tcp_pos
        delta = tgt - tcp
        n = np.linalg.norm(delta)
        if n > 0.04:  # anti-windup: the target never runs far ahead of the tool
            tgt = tcp + delta * 0.04 / n
        self.tcp_target = tgt
        self.yaw_target = wrap_pi(self.yaw_target + a[3] * self.yaw_scale)
        self.grip_cmd = FINGER_TRAVEL if a[4] > 0 else 0.0

        self.q_cmd = self.ik.solve(self.q_cmd, self.tcp_target, self.yaw_target)
        d.ctrl[:5] = self.q_cmd
        d.ctrl[5:7] = self.grip_cmd

        for _ in range(self.frame_skip):
            mujoco.mj_step(m, d)
        self.time += self.control_dt

        self._update_slip()
        self._update_machine()
        self.prev_step = self.step_now
        self.step_now = self._sop_step()

        reward = 0.0
        if self.step_now > self.stats.max_step:
            reward += float(self.step_now - self.stats.max_step)
            self.stats.max_step = int(self.step_now)
        if self.step_now < self.prev_step:
            self.stats.regressions += 1
            if not self.grasped:
                self.stats.drops += 1
            reward -= 1.0

        success = self.step_now == Step.DONE
        fell = self.part_pos[2] < -0.05
        terminated = bool(success or fell)
        truncated = bool(self.time >= self.max_time and not terminated)
        if success:
            reward += 5.0

        info = self._info()
        info["success"] = bool(success)
        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, terminated, truncated, info

    def needs_manual_fix(self) -> bool:
        """Part is lying on its side (or wedged) and not in the bin: a top-down gripper cannot recover it."""
        return self.part_upright < 0.9 and not self.grasped and not self._in_bin()

    def manual_fix(self):
        """The operator reaches into the cell and stands the part upright in the input tray.
        Keeps the part's finished/raw state. Callers charge the operator time."""
        rng, d = self.np_random, self.data
        xy = np.array([rng.uniform(0.40, 0.50), rng.uniform(-0.30, -0.20)])
        yaw = rng.uniform(-0.35, 0.35)
        a = self.part_qadr
        d.qpos[a:a + 3] = [xy[0], xy[1], PART_HALF_Z + 0.001]
        d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qvel[self.part_dadr:self.part_dadr + 6] = 0.0
        self.model.geom_friction[self.slip_geoms] = self.base_friction
        self.slip_until = None
        mujoco.mj_forward(self.model, d)
        self.stats.manual_fixes += 1
        self.step_now = self._sop_step()

    # ------------------------------------------------------------------ world state
    @property
    def tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.tcp_site].copy()

    @property
    def tcp_yaw(self) -> float:
        return mat_yaw(self.data.site_xmat[self.tcp_site].reshape(3, 3))

    @property
    def part_pos(self) -> np.ndarray:
        return self.data.xpos[self.part_body].copy()

    @property
    def part_yaw(self) -> float:
        return mat_yaw(self.data.xmat[self.part_body].reshape(3, 3))

    @property
    def part_upright(self) -> float:
        return float(self.data.xmat[self.part_body].reshape(3, 3)[2, 2])

    @property
    def fixture_pos(self) -> np.ndarray:
        return self.data.xpos[self.fixture_body].copy()

    @property
    def fixture_yaw(self) -> float:
        return mat_yaw(self.data.xmat[self.fixture_body].reshape(3, 3))

    @property
    def bin_pos(self) -> np.ndarray:
        return self.data.xpos[self.bin_body].copy()

    @property
    def gripper_opening(self) -> float:
        return float(1.0 - self.data.qpos[self.finger_qadr].mean() / FINGER_TRAVEL)

    @property
    def grasped(self) -> bool:
        touching = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if self.part_geom in pair:
                touching |= pair & self.finger_geoms
        return touching == self.finger_geoms and self.grip_cmd > 0

    def _surface_z(self, xy: np.ndarray) -> float:
        mxy = self.data.xpos[self.machine_body][:2]
        return MACHINE_TOP if np.all(np.abs(xy - mxy) < MACHINE_HALF + 0.02) else 0.0

    def _lifted(self) -> bool:
        p = self.part_pos
        return self.grasped and not self._seated(loose=True) and p[2] > self._surface_z(p[:2]) + PART_HALF_Z + 0.015

    def _seated(self, loose: bool = False) -> bool:
        p = self.part_pos
        tol = 0.012 if loose else 0.008
        return (np.linalg.norm(p[:2] - self.fixture_pos[:2]) < tol
                and abs(p[2] - SEAT_Z) < (0.025 if loose else 0.006)
                and self.part_upright > 0.98)

    def _in_bin(self) -> bool:
        p = self.part_pos
        return bool(np.all(np.abs(p[:2] - self.bin_pos[:2]) < BIN_INNER) and p[2] < 0.08)

    def _withdrawn(self) -> bool:
        return self.gripper_opening > 0.8 and self.tcp_pos[2] > self.part_pos[2] + 0.06

    def _sop_step(self) -> Step:
        if self.machining_started_at is not None and not self.finished:
            return Step.MACHINING
        if not self.finished:
            if self._seated() and self._withdrawn():
                return Step.MACHINING
            if self._seated(loose=True) or self._lifted():
                return Step.LOAD
            return Step.PICK
        if self._in_bin() and not self.grasped and self._withdrawn():
            return Step.DONE
        if self._lifted() or self._in_bin():
            return Step.PLACE
        return Step.UNLOAD

    def _update_machine(self):
        m = self.model
        if self.machining_started_at is None and not self.finished and self._seated() and self._withdrawn():
            self.machining_started_at = self.time
            m.geom_rgba[self.light_geom] = LIGHT_RUN
        if self.machining_started_at is not None and not self.finished:
            if self.time - self.machining_started_at >= self.cycle_time:
                self.finished = True
                m.geom_rgba[self.part_geom] = DONE_RGBA
                m.geom_rgba[self.light_geom] = LIGHT_IDLE

    def _update_slip(self):
        if "slip" not in self.stats.faults:
            return
        carrying = self.step_now in (Step.LOAD, Step.PLACE) and self._lifted()
        if self.slip_at is None and carrying:
            if self.carry_started_at is None:
                self.carry_started_at = self.time
                self.slip_trigger_delay = self.np_random.uniform(0.3, 1.2)
            elif self.time - self.carry_started_at > self.slip_trigger_delay:
                self.slip_at = self.time
                self.slip_until = self.time + 0.6
                # MuJoCo combines contact friction with max(), so every geom in the grip pair must slip.
                self.model.geom_friction[self.slip_geoms] = [0.02, 0.0, 0.0]
        if self.slip_until is not None and self.time >= self.slip_until:
            self.model.geom_friction[self.slip_geoms] = self.base_friction
            self.slip_until = None

    # ------------------------------------------------------------------ obs / info
    def _obs(self) -> np.ndarray:
        tcp = self.tcp_pos
        tyaw = self.tcp_yaw
        onehot = np.zeros(6)
        onehot[int(self.step_now)] = 1.0
        prog = 0.0
        if self.machining_started_at is not None:
            prog = min(1.0, (self.time - self.machining_started_at) / self.cycle_time)
        obs = np.concatenate([
            tcp,                                                     # 0:3
            [np.sin(tyaw), np.cos(tyaw)],                            # 3:5
            [self.gripper_opening],                                  # 5
            self.part_pos - tcp,                                     # 6:9
            [wrap_quarter(self.part_yaw - tyaw)],                    # 9
            self.fixture_pos - tcp,                                  # 10:13
            [wrap_quarter(self.fixture_yaw - tyaw)],                 # 13
            self.bin_pos - tcp,                                      # 14:17
            [float(self.grasped)],                                   # 17
            [float(self.finished)],                                  # 18
            onehot,                                                  # 19:25
            [prog],                                                  # 25
            self.tcp_target - tcp,                                   # 26:29 controller lag
            *self._goal_features(),                                  # 29:37 task frame
        ])
        return obs.astype(np.float32)

    def _goal_features(self):
        """Vector from the thing being moved to where the current SOP step wants it, in metres,
        plus a tanh-compressed copy with ~1 cm resolution so millimetre errors are visible to a policy."""
        tcp, p, step = self.tcp_pos, self.part_pos, self.step_now
        if step in (Step.PICK, Step.UNLOAD):
            d = p + np.array([0, 0, GRASP_OFFSET]) - tcp
            dyaw = wrap_quarter(self.part_yaw - self.tcp_yaw)
        elif step == Step.LOAD:
            d = np.r_[self.fixture_pos[:2], SEAT_Z] - p
            dyaw = wrap_quarter(self.fixture_yaw - self.part_yaw)
        elif step == Step.PLACE:
            d = np.r_[self.bin_pos[:2], 0.035] - p
            dyaw = 0.0
        else:
            d = self.home_pos - tcp
            dyaw = 0.0
        return [d, np.tanh(d / 0.01), [dyaw, np.tanh(dyaw / 0.03)]]

    def _info(self) -> dict:
        return {
            "step": int(self.step_now),
            "step_name": STEP_NAMES[int(self.step_now)],
            "time": self.time,
            "tcp_pos": self.tcp_pos,
            "tcp_target": self.tcp_target.copy(),
            "tcp_yaw": self.tcp_yaw,
            "yaw_target": self.yaw_target,
            "gripper_opening": self.gripper_opening,
            "grasped": self.grasped,
            "part_pos": self.part_pos,
            "part_yaw": self.part_yaw,
            "fixture_pos": self.fixture_pos,
            "fixture_yaw": self.fixture_yaw,
            "bin_pos": self.bin_pos,
            "finished": self.finished,
            "seated": self._seated(),
            "regressions": self.stats.regressions,
            "drops": self.stats.drops,
            "manual_fixes": self.stats.manual_fixes,
            "part_upright": self.part_upright,
            "needs_manual_fix": self.needs_manual_fix(),
            "max_step": self.stats.max_step,
            "faults": list(self.stats.faults),
        }

    # ------------------------------------------------------------------ rendering
    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, self.height, self.width)
            self._renderer.update_scene(self.data, camera=self.camera)
            return self._renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer  # on macOS this requires launching the script with `mjpython`
                self._viewer = viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
