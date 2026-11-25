"""The ask-assist-learn loop in simulation: demos -> deploy with asking -> operator corrections -> retrain, repeated.

Reports, per round: autonomous success (no help allowed), and for the deployed HITL cell:
success, asks per episode, operator minutes per robot-hour, autonomy fraction, parts per hour.
Baselines: always-teleoperate, and the round-0 policy with no help.

    python scripts/run_loop.py                      # clean demos, hard-fault deployment, 4 rounds
    python scripts/run_loop.py --robot-weight 0     # pure HG-DAgger instead of Sirius weighting
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aal_sim import FaultConfig, MachineTendingEnv, ScriptedOperator  # noqa: E402
from aal_sim.hitl import Gate, OperatorModel, run_episode, sirius_weights, summarize  # noqa: E402
from aal_sim.policy import EnsemblePolicy  # noqa: E402

COLS = ["success_rate", "asks_per_episode", "operator_min_per_robot_hour", "autonomy_fraction",
        "parts_per_hour", "regressions_per_episode"]


def fmt_row(name: str, s: dict, auto: float | None = None) -> str:
    a = f"{auto:6.0%}" if auto is not None else "     -"
    return (f"{name:<22} {a} {s['success_rate']:7.0%} {s['asks_per_episode']:6.2f} "
            f"{s['operator_min_per_robot_hour']:8.1f} {s['autonomy_fraction']:8.0%} "
            f"{s['parts_per_hour']:7.0f} {s['regressions_per_episode']:6.2f}")


HEADER = (f"{'':<22} {'auto':>6} {'success':>7} {'asks':>6} {'op-min/h':>8} {'autonomy':>8} "
          f"{'parts/h':>7} {'regr':>6}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faults", default="hard", choices=["none", "mild", "hard"], help="deployment faults")
    ap.add_argument("--demo-faults", default="none", choices=["none", "mild", "hard"],
                    help="faults while collecting demos (default: clean demos, faulty deployment)")
    ap.add_argument("--demos", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=20, help="HITL deployment episodes per round")
    ap.add_argument("--eval", type=int, default=30, help="autonomous eval episodes per round")
    ap.add_argument("--fit-steps", type=int, default=3000)
    ap.add_argument("--operator-weight", type=float, default=2.0)
    ap.add_argument("--robot-weight", type=float, default=1.0,
                    help="weight of autonomous samples when retraining (1 = Sirius, 0 = HG-DAgger)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.12)
    ap.add_argument("--no-sop-aware", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    import torch
    if args.threads:
        torch.set_num_threads(args.threads)
    out = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    env = MachineTendingEnv(faults=FaultConfig.preset(args.faults))
    operator = ScriptedOperator()
    op_model = OperatorModel()
    gate = Gate(threshold=args.threshold, sop_aware=not args.no_sop_aware)
    seeds = iter(range(args.seed * 100_000, args.seed * 100_000 + 100_000))
    eval_seeds = list(range(10_000_000, 10_000_000 + args.eval))

    def evaluate(policy):
        logs = [run_episode(env, "autonomous", policy=policy, seed=s) for s in eval_seeds]
        return summarize(logs)

    t0 = time.time()
    print(f"faults={args.faults}  demo_faults={args.demo_faults}  robot_w={args.robot_weight}  demos={args.demos}  rounds={args.rounds}x{args.episodes}  out={out}")

    # Baseline: always teleoperate.
    teleop = summarize([run_episode(env, "teleop", operator=operator, seed=s) for s in eval_seeds])

    # Round 0: behavior cloning on demonstrations (DART-style noise for state coverage).
    demo_opts = {"faults": FaultConfig.preset(args.demo_faults)}
    demo_logs = [run_episode(env, "teleop", operator=operator, seed=next(seeds), demo_noise=0.3, options=demo_opts)
                 for _ in range(args.demos)]
    O, A, W = sirius_weights(demo_logs, operator_w=1.0)
    data = [(O, A, W)]
    torch.manual_seed(args.seed)
    policy = EnsemblePolicy(env.obs_dim, 5)
    loss = policy.fit(O, A, W, steps=args.fit_steps, seed=args.seed)
    print(f"round 0: {len(O)} demo samples, fit loss {loss:.4f}  [{time.time() - t0:.0f}s]")
    policy.save(out / "policy_r0.pt")

    rows, history = [], []
    for r in range(args.rounds + 1):
        auto = evaluate(policy)
        if r == 0:
            rows.append(("no help (BC only)", auto, auto["success_rate"]))
        if r == args.rounds:
            final_auto = auto
            history.append({"round": r, "autonomous": auto})
            break
        logs = [run_episode(env, "hitl", policy=policy, operator=operator, gate=gate, op_model=op_model,
                            seed=next(seeds)) for _ in range(args.episodes)]
        dep = summarize(logs)
        rows.append((f"round {r} (deployed)", dep, auto["success_rate"]))
        history.append({"round": r, "autonomous": auto, "deployed": dep})
        print(f"round {r}: auto {auto['success_rate']:.0%}  deployed {dep['success_rate']:.0%}  "
              f"asks/ep {dep['asks_per_episode']:.2f} {dep['ask_reasons']}  "
              f"op-min/h {dep['operator_min_per_robot_hour']:.1f}  [{time.time() - t0:.0f}s]")
        data.append(sirius_weights(logs, operator_w=args.operator_weight, robot_w=args.robot_weight))
        O, A, W = (np.concatenate(x) for x in zip(*data))
        torch.manual_seed(args.seed * 1000 + r + 1)
        policy = EnsemblePolicy(env.obs_dim, 5)
        policy.fit(O, A, W, steps=args.fit_steps, seed=args.seed + r + 1)
        policy.save(out / f"policy_r{r + 1}.pt")

    print()
    print(HEADER)
    print(fmt_row("always teleoperate", teleop))
    for name, s, a in rows:
        print(fmt_row(name, s, a))
    print(f"{'final policy, no help':<22} {final_auto['success_rate']:6.0%}")
    print(f"\nauto = autonomous success on {args.eval} held-out episodes with no help allowed.")

    json.dump({"args": vars(args), "teleop": teleop, "history": history}, open(out / "metrics.json", "w"),
              indent=2, default=float)
    print(f"saved {out}/metrics.json  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
