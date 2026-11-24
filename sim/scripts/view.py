"""Watch the cell live in the MuJoCo viewer, in real time.

macOS needs the viewer on the main thread, so launch with `mjpython` (installed with mujoco):
    .venv/bin/mjpython scripts/view.py --mode teleop --faults hard
    .venv/bin/mjpython scripts/view.py --mode hitl --policy runs/<run>/policy_r4.pt --faults hard
Linux/Windows: plain `python` works.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mujoco import viewer  # noqa: E402

from aal_sim import MachineTendingEnv, ScriptedOperator  # noqa: E402
from aal_sim.hitl import OPERATOR, Gate, run_episode  # noqa: E402
from aal_sim.policy import EnsemblePolicy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="teleop", choices=["teleop", "autonomous", "hitl"])
    ap.add_argument("--policy")
    ap.add_argument("--faults", default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed vs real time")
    args = ap.parse_args()

    env = MachineTendingEnv(faults=args.faults)
    policy = EnsemblePolicy.load(args.policy) if args.policy else None
    if args.mode != "teleop" and policy is None:
        ap.error("--policy is required for autonomous/hitl")

    with viewer.launch_passive(env.model, env.data) as v:
        state = {"who": None, "step": None}

        def cb(env, info, who, unc):
            if who != state["who"] or info["step"] != state["step"]:
                print(f"t={info['time']:5.1f}s  {info['step_name']:<9}  {'OPERATOR' if who == OPERATOR else 'robot'}"
                      f"  unc={unc:.3f}")
                state.update(who=who, step=info["step"])
            v.sync()
            time.sleep(env.control_dt / args.speed)
            if not v.is_running():
                raise SystemExit

        seed = args.seed
        while v.is_running():
            log = run_episode(env, args.mode, policy=policy, operator=ScriptedOperator(), gate=Gate(),
                              seed=seed, frame_cb=cb)
            print(f"episode seed={seed}: success={log.success} faults={log.faults} asks={len(log.asks)}\n")
            seed += 1


if __name__ == "__main__":
    main()
