# Ask, Assist, Learn: MuJoCo cell

A simulated machine-tending cell for the thesis loop: **run the SOP → ask for help → operator assists → log the correction → retrain**.

A 5-DOF arm with a parallel gripper tends a CNC machine:

| SOP step | What has to happen |
|---|---|
| PICK | grasp the raw part from the input tray and lift it |
| LOAD | seat it in the machine's fixture (6 mm clearance per side, yaw-aligned), release, withdraw |
| MACHINING | 3 s cycle; the status light turns amber and the part changes colour when done |
| UNLOAD | grasp the finished part and lift it out of the fixture |
| PLACE | drop it in the output bin and withdraw |

The current step is derived from world state every control step. If the robot drops a part, the step falls back (e.g. LOAD → PICK) and the drop counts as a regression.

## Setup

```bash
cd sim
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q tests
```

## Run it

```bash
# The full loop: 20 clean demos, then 4 rounds of deployment into a cell with faults
.venv/bin/python scripts/run_loop.py

# Video with an overlay showing the SOP step, who is in control, and the ask signal
.venv/bin/python scripts/record_video.py --mode teleop --force-faults slip --out cell.mp4
.venv/bin/python scripts/record_video.py --mode hitl --policy runs/<run>/policy_r0.pt --faults hard --out hitl.mp4

# Live viewer (macOS needs mjpython, which ships with mujoco)
.venv/bin/mjpython scripts/view.py --mode hitl --policy runs/<run>/policy_r4.pt --faults hard
```

As a Gymnasium env:

```python
import gymnasium as gym, aal_sim
env = gym.make("AskAssistLearn/MachineTending-v0", faults="hard")
obs, info = env.reset(seed=0)            # obs: (37,), info: privileged state + SOP step
obs, r, terminated, truncated, info = env.step(env.action_space.sample())
```

## What's in it

| File | Role |
|---|---|
| `aal_sim/assets/cell.xml` | MJCF: arm, gripper, machine, fixture pocket, tray, bin, cameras (`overview`, `top`, `wrist`) |
| `aal_sim/env.py` | Gymnasium env, SOP step logic, fault injection, machining cycle, rewards |
| `aal_sim/ik.py` | Damped least-squares IK for a top-down tool pose (position and yaw) |
| `aal_sim/expert.py` | Scripted operator: a privileged state machine that finishes every step and recovers from drops |
| `aal_sim/policy.py` | Bootstrap ensemble of BC MLPs; ensemble disagreement is the uncertainty signal |
| `aal_sim/hitl.py` | Ask gate, operator time model, episode runner, metrics, Sirius-style sample weights |
| `scripts/run_loop.py` | Demos → deploy → retrain rounds, with baselines |

**Control.** Action is `(dx, dy, dz, dyaw, grip)` in [-1, 1] at 20 Hz, up to 15 mm and 0.12 rad per step. IK solves joint targets on a scratch `MjData`, and position actuators with gravity compensation track them. Physics runs at 500 Hz.

**Observation (37).** Tool pose, gripper opening, part / fixture / bin poses relative to the tool, a grasp flag, a finished flag, a one-hot SOP step, machining progress, and controller lag. It also includes a task-frame block: the vector from the held object to the current step's goal, plus a `tanh(d / 1 cm)` copy. Without that block BC cannot see millimetre-level fixture errors and stalls above the fixture (0% success in testing).

**Faults** (`faults="none" | "mild" | "hard"`, or `options={"force_faults": [...]}` at reset):

| Fault | Effect | mild | hard |
|---|---|---|---|
| `ood_pose` | raw part arrives anywhere in a 30 × 34 cm area, at any yaw | 15% | 35% |
| `fixture_shift` | fixture offset up to ±25 mm and ±17° | 15% | 35% |
| `slip` | grip friction collapses for 0.6 s mid-carry, so the part drops | 10% | 25% |

A part that lands on its side can't be recovered by a top-down gripper. The cell flags `needs_manual_fix`, and the operator walks in and stands it back up in the tray (charged at 20 s of operator time).

**When the robot asks** (`hitl.Gate`):
- Ensemble disagreement above a threshold for 3 consecutive steps. The threshold is 40% lower on LOAD, the risky step (`sop_aware`).
- No SOP progress for 10 s.
- A tipped part.

The operator responds after 3 s, drives until the current SOP step is done (15 s cap), then hands back control.

**Retraining** (`hitl.sirius_weights`) weights the logged samples by source:
- operator corrections: `--operator-weight` (default 2)
- robot samples in the 1.5 s before an ask: 0
- other autonomous samples: `--robot-weight` (1 = Sirius, 0 = pure HG-DAgger)

## Results so far

`scripts/run_loop.py` defaults: 20 clean demos, hard-fault deployment, 20 deployed episodes per round, 30 held-out autonomous eval episodes, seed 0. Takes about 7 minutes on an M1 Pro.

| | auto | deployed success | asks/ep | operator min / robot-hour | autonomy | parts/h |
|---|---|---|---|---|---|---|
| always teleoperate | – | 100% | – | 60.0 | 0% | 214 |
| BC only, no help | 83% | 83% | – | 0 | 100% | 167 |
| round 0 | 83% | 100% | 1.95 | 21.4 | 78% | 142 |
| round 1 | 77% | 100% | 0.15 | 11.0 | 100% | 220 |
| round 2 | 80% | 100% | 0.40 | 16.3 | 98% | 187 |
| round 3 | 77% | 100% | 0.35 | 11.0 | 99% | 214 |

*auto* is success with no help allowed, on held-out seeds.

What this shows, and what it doesn't:
- Asking keeps the deployed cell at 100% while cutting operator time from 60 to about 11 min per robot-hour.
- Part of the remaining operator time is irreducible: tipped parts after slips need a person.
- **Autonomous success does not climb across rounds yet.** The remaining failures are the policy getting stuck after slips and on out-of-distribution poses. Being stuck is a low-uncertainty state (the members agree on doing nothing), so the uncertainty gate misses it and only the 10 s stall timer catches it. That is exactly the case for a learned failure predictor over novelty gating.
- These are single-seed numbers on 30 eval episodes (±8% binomial noise). Run several `--seed` values before reading anything into round-to-round changes.

## Not here yet

- The humanoid (G1 / R1) version of the cell.
- A learned failure predictor for the gate; it would train on the logged `EpisodeLog`s.
- RL fine-tuning of the LOAD step (residual RL or DPPO).
- A VLA comparison.
- Image observations. The `wrist` and `overview` cameras exist, but the policies use state only.
- Real VR teleop; the operator here is scripted. The response-delay and takeover times in `OperatorModel` are guesses until measured on hardware.
