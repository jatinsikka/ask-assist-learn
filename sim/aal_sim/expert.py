"""Scripted operator: a privileged state machine that can finish every SOP step, including recovery
from drops. It produces demonstrations and plays the human operator during takeovers."""

from __future__ import annotations

import numpy as np

from .env import GRASP_OFFSET, SEAT_Z, MachineTendingEnv, Step, wrap_pi, wrap_quarter

CARRY_Z = 0.23        # tcp height while carrying (clears the machine and the bin walls)
HOVER = 0.08          # hover above the grasp height before descending
HOME = MachineTendingEnv.home_pos


class ScriptedOperator:
    def __init__(self, noise: float = 0.0, rng: np.random.Generator | None = None):
        self.noise = noise
        self.rng = rng or np.random.default_rng()
        self.reset()

    def reset(self):
        self.retry = False      # last grasp closed on nothing: open and re-approach
        self.close_steps = 0

    # ------------------------------------------------------------------
    def act(self, info: dict) -> np.ndarray:
        step = Step(info["step"])
        if step in (Step.PICK, Step.UNLOAD):
            pos, yaw, close = self._grasp(info)
        elif step == Step.LOAD:
            pos, yaw, close = self._insert(info)
        elif step == Step.PLACE:
            pos, yaw, close = self._place(info)
        else:  # MACHINING or DONE: wait at home with the gripper open
            pos, yaw, close = HOME.copy(), info["tcp_yaw"], False
        if step not in (Step.PICK, Step.UNLOAD):
            self.close_steps, self.retry = 0, False
        return self._to_action(info, pos, yaw, close)

    def _to_action(self, info, pos, yaw, close) -> np.ndarray:
        dp = (pos - info["tcp_target"]) / MachineTendingEnv.pos_scale
        dyaw = wrap_pi(yaw - info["yaw_target"]) / MachineTendingEnv.yaw_scale
        a = np.concatenate([dp, [dyaw], [1.0 if close else -1.0]])
        if self.noise > 0:
            a[:4] += self.rng.normal(0, self.noise, 4)
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    def _grasp(self, info):
        tcp, p = info["tcp_pos"], info["part_pos"]
        yaw = info["tcp_yaw"] + wrap_quarter(info["part_yaw"] - info["tcp_yaw"])
        grasp_z = p[2] + GRASP_OFFSET
        hover_z = grasp_z + HOVER

        if info["grasped"]:
            self.close_steps = 0
            return np.array([tcp[0], tcp[1], grasp_z + 0.12]), info["tcp_yaw"], True

        xy_err = np.linalg.norm(p[:2] - tcp[:2])
        yaw_err = abs(wrap_quarter(info["part_yaw"] - info["tcp_yaw"]))

        if self.retry:
            if tcp[2] > hover_z - 0.01 and info["gripper_opening"] > 0.9:
                self.retry = False
            return np.array([tcp[0], tcp[1], hover_z]), yaw, False

        at_grasp = xy_err < 0.008 and tcp[2] < grasp_z + 0.006
        if at_grasp:
            self.close_steps += 1
            if info["gripper_opening"] < 0.08 or self.close_steps > 25:   # closed on nothing
                self.retry, self.close_steps = True, 0
                return np.array([tcp[0], tcp[1], hover_z]), yaw, False
            return np.array([p[0], p[1], grasp_z]), yaw, True
        self.close_steps = 0

        if xy_err > 0.006 or yaw_err > 0.05:
            if tcp[2] < grasp_z + 0.04 and xy_err > 0.015:          # too low to slide over: rise first
                return np.array([tcp[0], tcp[1], hover_z]), yaw, False
            z = hover_z if xy_err > 0.03 else max(tcp[2] - 0.0, grasp_z + 0.01)
            return np.array([p[0], p[1], z]), yaw, False
        return np.array([p[0], p[1], grasp_z]), yaw, False

    def _insert(self, info):
        tcp, p, f = info["tcp_pos"], info["part_pos"], info["fixture_pos"]
        if not info["grasped"]:  # released in the pocket (or lost it): back off upward
            return np.array([tcp[0], tcp[1], p[2] + 0.12]), info["tcp_yaw"], False

        offset = tcp[:2] - p[:2]                     # where the part sits in the grasp
        goal_xy = f[:2] + offset
        xy_err = np.linalg.norm(f[:2] - p[:2])
        dyaw = wrap_quarter(info["fixture_yaw"] - info["part_yaw"])
        yaw = info["tcp_yaw"] + dyaw
        clear_z = SEAT_Z + 0.045 + GRASP_OFFSET      # part bottom above the pocket walls

        if xy_err > 0.03:
            if tcp[2] < CARRY_Z - 0.03 and np.linalg.norm(goal_xy - tcp[:2]) > 0.05:
                return np.array([tcp[0], tcp[1], CARRY_Z]), yaw, True
            return np.array([goal_xy[0], goal_xy[1], CARRY_Z]), yaw, True
        if xy_err > 0.003 or abs(dyaw) > 0.025:
            return np.array([goal_xy[0], goal_xy[1], max(tcp[2], clear_z)]), yaw, True
        seat_tcp_z = SEAT_Z + (tcp[2] - p[2]) - 0.002
        if p[2] < SEAT_Z + 0.004:
            return np.array([goal_xy[0], goal_xy[1], seat_tcp_z]), yaw, False   # seated: release
        return np.array([goal_xy[0], goal_xy[1], seat_tcp_z]), yaw, True

    def _place(self, info):
        tcp, p, b = info["tcp_pos"], info["part_pos"], info["bin_pos"]
        if not info["grasped"]:
            return np.array([tcp[0], tcp[1], max(p[2] + 0.12, tcp[2])]), info["tcp_yaw"], False
        offset = tcp[:2] - p[:2]
        goal_xy = b[:2] + offset
        xy_err = np.linalg.norm(b[:2] - p[:2])
        if xy_err > 0.03:
            if tcp[2] < CARRY_Z - 0.03 and np.linalg.norm(goal_xy - tcp[:2]) > 0.05:
                return np.array([tcp[0], tcp[1], CARRY_Z]), info["tcp_yaw"], True
            return np.array([goal_xy[0], goal_xy[1], CARRY_Z]), info["tcp_yaw"], True
        release_z = 0.005 + 0.03 + (tcp[2] - p[2]) + 0.012
        if tcp[2] < release_z + 0.006:
            return np.array([goal_xy[0], goal_xy[1], release_z]), info["tcp_yaw"], False
        return np.array([goal_xy[0], goal_xy[1], release_z]), info["tcp_yaw"], True
