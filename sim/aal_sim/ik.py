"""Damped least-squares IK for a top-down tool pose (position + yaw, tool z pointing down)."""

from __future__ import annotations

import mujoco
import numpy as np

ARM_JOINTS = ("j1", "j2", "j3", "j4", "j5")


def yaw_matrix(yaw: float) -> np.ndarray:
    """Tool orientation: tool z points down (world -z), tool x rotated by yaw about world z."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class TopDownIK:
    """Solves joint targets on a scratch MjData so the solve never disturbs the simulation."""

    def __init__(self, model: mujoco.MjModel, site: str = "tcp", damping: float = 0.05):
        self.model = model
        self.scratch = mujoco.MjData(model)
        self.site_id = model.site(site).id
        self.qadr = np.array([model.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.dadr = np.array([model.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.lo = model.jnt_range[[model.joint(j).id for j in ARM_JOINTS], 0]
        self.hi = model.jnt_range[[model.joint(j).id for j in ARM_JOINTS], 1]
        self.damping = damping
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = self.scratch
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, d)
        return d.site_xpos[self.site_id].copy(), d.site_xmat[self.site_id].reshape(3, 3).copy()

    def solve(self, q0: np.ndarray, pos: np.ndarray, yaw: float, iters: int = 4,
              max_step: float = 0.25) -> np.ndarray:
        q = q0.copy()
        Rd = yaw_matrix(yaw)
        for _ in range(iters):
            p, R = self.fk(q)
            ep = pos - p
            er = 0.5 * sum(np.cross(R[:, i], Rd[:, i]) for i in range(3))
            if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
                break
            mujoco.mj_comPos(self.model, self.scratch)
            mujoco.mj_jacSite(self.model, self.scratch, self._jacp, self._jacr, self.site_id)
            J = np.vstack([self._jacp[:, self.dadr], 0.5 * self._jacr[:, self.dadr]])
            e = np.concatenate([ep, 0.5 * er])
            dq = J.T @ np.linalg.solve(J @ J.T + self.damping**2 * np.eye(6), e)
            n = np.abs(dq).max()
            if n > max_step:
                dq *= max_step / n
            q = np.clip(q + dq, self.lo, self.hi)
        return q
