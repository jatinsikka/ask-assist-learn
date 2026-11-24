"""Run -> ask -> assist -> learn.

`run_episode` deploys a policy in the cell. A gate decides when the robot asks for help; the
operator (scripted here, a VR teleoperator on hardware) responds after a delay, takes over until
the current SOP step is done, then hands control back. Every sample is logged with who produced it,
so retraining can weight operator corrections (HG-DAgger with Sirius-style weights).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .env import MachineTendingEnv, Step
from .expert import ScriptedOperator
from .policy import EnsemblePolicy

ROBOT, OPERATOR = 0, 1


@dataclass
class Gate:
    """When should the robot ask for help?

    - uncertainty: ensemble disagreement above a threshold (lower on risky SOP steps when sop_aware)
    - stall: no SOP progress for `stall_s` seconds
    - unrecoverable: the cell sees the part tipped over (needs a person)
    """
    threshold: float = 0.12
    risky_scale: float = 0.6          # LOAD threshold multiplier when sop_aware
    sop_aware: bool = True
    stall_s: float = 10.0
    patience: int = 3                 # consecutive uncertain steps before asking

    def __post_init__(self):
        self._count = 0

    def reset(self):
        self._count = 0

    def should_ask(self, unc: float, info: dict, since_progress: float) -> str | None:
        if info["needs_manual_fix"]:
            return "unrecoverable"
        if since_progress > self.stall_s:
            return "stall"
        thr = self.threshold
        if self.sop_aware and info["step"] == Step.LOAD:
            thr *= self.risky_scale
        self._count = self._count + 1 if unc > thr else 0
        if self._count >= self.patience:
            return "uncertain"
        return None


@dataclass
class OperatorModel:
    """Time accounting for the human. All times in seconds of simulated time."""
    response_delay_s: float = 3.0     # robot holds still until the operator picks up
    max_takeover_s: float = 15.0
    manual_fix_s: float = 20.0        # walk in, stand the part up, walk out


@dataclass
class EpisodeLog:
    obs: list = field(default_factory=list)
    act: list = field(default_factory=list)
    who: list = field(default_factory=list)
    unc: list = field(default_factory=list)
    step: list = field(default_factory=list)
    asks: list = field(default_factory=list)          # (time, reason, step)
    success: bool = False
    robot_time: float = 0.0           # wall-clock time of the cell (sim time incl. waits)
    operator_time: float = 0.0        # time the operator was busy (teleop + manual fixes)
    idle_time: float = 0.0            # robot waiting for the operator to respond
    regressions: int = 0
    drops: int = 0
    faults: list = field(default_factory=list)
    max_step: int = 0

    def arrays(self):
        return (np.asarray(self.obs, np.float32), np.asarray(self.act, np.float32),
                np.asarray(self.who, np.int8))


def run_episode(env: MachineTendingEnv, mode: str, policy: EnsemblePolicy | None = None,
                operator: ScriptedOperator | None = None, gate: Gate | None = None,
                op_model: OperatorModel | None = None, seed: int | None = None,
                demo_noise: float = 0.0, frame_cb=None, options: dict | None = None) -> EpisodeLog:
    """mode: 'teleop' (operator only), 'autonomous' (policy only, never asks), 'hitl' (policy + gate + operator)."""
    assert mode in ("teleop", "autonomous", "hitl")
    operator = operator or ScriptedOperator()
    gate = gate or Gate()
    op_model = op_model or OperatorModel()
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed, options=options)
    operator.reset()
    gate.reset()
    log = EpisodeLog(faults=list(info["faults"]))

    in_control = OPERATOR if mode == "teleop" else ROBOT
    takeover_started, takeover_step = 0.0, None
    best_step, last_progress = info["step"], 0.0
    pending_ask = None

    while True:
        t = info["time"]
        if info["step"] > best_step:
            best_step, last_progress = info["step"], t

        # Unrecoverable states need a person in the cell, regardless of who is driving.
        if info["needs_manual_fix"] and mode != "autonomous":
            if mode == "hitl":
                log.asks.append((t, "unrecoverable", info["step"]))
                log.idle_time += op_model.response_delay_s
            log.operator_time += op_model.manual_fix_s
            log.robot_time += op_model.response_delay_s * (mode == "hitl") + op_model.manual_fix_s
            env.manual_fix()
            info = env._info()
            obs = env._obs()
            operator.reset()
            last_progress = t
            continue

        if in_control == ROBOT:
            a, unc = policy.act(obs)
            reason = gate.should_ask(unc, info, t - last_progress) if mode == "hitl" else None
            if reason is not None:
                log.asks.append((t, reason, info["step"]))
                # Robot holds position while the operator picks up.
                log.idle_time += op_model.response_delay_s
                log.robot_time += op_model.response_delay_s
                in_control = OPERATOR
                takeover_started, takeover_step = t, info["step"]
                operator.reset()
                gate.reset()
                continue
            who = ROBOT
        else:
            a = operator.act(info)
            unc = 0.0
            who = OPERATOR
            if mode == "hitl":
                if info["step"] > takeover_step or t - takeover_started > op_model.max_takeover_s:
                    in_control = ROBOT
                    last_progress = t
                    continue

        log.obs.append(obs)
        log.act.append(a)
        log.who.append(who)
        log.unc.append(unc)
        log.step.append(info["step"])

        exec_a = a
        if who == OPERATOR and demo_noise > 0 and mode == "teleop":
            exec_a = np.clip(a + np.r_[rng.normal(0, demo_noise, 4), 0.0], -1, 1)
        obs, _, term, trunc, info = env.step(exec_a)
        log.robot_time += env.control_dt
        if who == OPERATOR:
            log.operator_time += env.control_dt
        if frame_cb is not None:
            frame_cb(env, info, who, unc)
        if term or trunc:
            break

    log.success = bool(info["success"])
    log.regressions = info["regressions"]
    log.drops = info["drops"]
    log.max_step = info["max_step"]
    return log


def summarize(logs: list[EpisodeLog]) -> dict:
    robot_h = sum(l.robot_time for l in logs) / 3600
    asks = [a for l in logs for a in l.asks]
    reasons = {}
    for _, r, _ in asks:
        reasons[r] = reasons.get(r, 0) + 1
    by_step = {}
    for _, _, s in asks:
        n = Step(s).name
        by_step[n] = by_step.get(n, 0) + 1
    n_samples = sum(len(l.who) for l in logs)
    return {
        "episodes": len(logs),
        "success_rate": float(np.mean([l.success for l in logs])),
        "asks_per_episode": len(asks) / max(1, len(logs)),
        "ask_reasons": reasons,
        "asks_by_step": by_step,
        "operator_min_per_robot_hour": (sum(l.operator_time for l in logs) / 60) / max(robot_h, 1e-9),
        "autonomy_fraction": 1 - sum(int(np.sum(np.asarray(l.who) == OPERATOR)) for l in logs) / max(1, n_samples),
        "mean_cycle_time_s": float(np.mean([l.robot_time for l in logs if l.success])) if any(l.success for l in logs) else float("nan"),
        "regressions_per_episode": float(np.mean([l.regressions for l in logs])),
        "parts_per_hour": sum(l.success for l in logs) / max(robot_h, 1e-9),
    }


def sirius_weights(logs: list[EpisodeLog], operator_w: float = 2.0, robot_w: float = 1.0,
                   pre_window_s: float = 1.5,
                   control_dt: float = MachineTendingEnv.control_dt):
    """Build (obs, act, weight) for retraining.

    operator samples: weight operator_w (corrections carry the most signal)
    robot samples right before an ask: weight 0 (they led to the need for help)
    other robot samples: weight robot_w (1 = Sirius, 0 = pure HG-DAgger)
    """
    O, A, W = [], [], []
    win = int(round(pre_window_s / control_dt))
    for l in logs:
        if not l.obs:
            continue
        o, a, who = l.arrays()
        w = np.where(who == OPERATOR, operator_w, robot_w)
        starts = np.flatnonzero((who[1:] == OPERATOR) & (who[:-1] == ROBOT)) + 1
        for s in starts:
            w[max(0, s - win):s] = 0.0
        O.append(o)
        A.append(a)
        W.append(w)
    return np.concatenate(O), np.concatenate(A), np.concatenate(W)
