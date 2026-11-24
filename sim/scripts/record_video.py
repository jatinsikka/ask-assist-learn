"""Record an MP4 of the cell. The overlay shows the SOP step, who is in control, and the ask signal.

    python scripts/record_video.py --mode teleop --out cell.mp4
    python scripts/record_video.py --mode hitl --policy runs/<run>/policy_r0.pt --faults hard --seed 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aal_sim import MachineTendingEnv, ScriptedOperator  # noqa: E402
from aal_sim.env import STEP_NAMES  # noqa: E402
from aal_sim.hitl import OPERATOR, Gate, run_episode  # noqa: E402
from aal_sim.policy import EnsemblePolicy  # noqa: E402

AMBER, TEAL, INK = (224, 161, 0), (12, 122, 114), (21, 25, 28)


def font(size):
    for name in ("Menlo.ttc", "DejaVuSansMono.ttf", "Courier New.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def overlay(frame, info, who, unc, gate_thr, fnt):
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im, "RGBA")
    W, H = im.size
    color = AMBER if who == OPERATOR else TEAL
    d.rectangle([0, 0, W - 1, H - 1], outline=color, width=6)
    d.rectangle([0, 0, W, 34], fill=(255, 255, 255, 215))
    seg = (W - 20) / 5
    for i, name in enumerate(STEP_NAMES[:5]):
        x0 = 10 + i * seg
        on = i == info["step"]
        fill = color + (255,) if on else ((205, 211, 207, 255) if i > info["step"] else (150, 160, 158, 255))
        d.rectangle([x0 + 2, 6, x0 + seg - 2, 28], fill=fill)
        d.text((x0 + 8, 9), name, fill=(255, 255, 255) if on else INK, font=fnt)
    label = "OPERATOR" if who == OPERATOR else "ROBOT"
    d.rectangle([0, H - 30, W, H], fill=(255, 255, 255, 215))
    d.text((10, H - 24), f"{label:<9} t={info['time']:5.1f}s", fill=color, font=fnt)
    if who != OPERATOR and gate_thr:
        bw = 160
        x0 = W - bw - 70
        d.text((x0 - 40, H - 24), "ask", fill=INK, font=fnt)
        d.rectangle([x0, H - 22, x0 + bw, H - 10], outline=INK)
        frac = min(1.0, unc / (2 * gate_thr))
        d.rectangle([x0, H - 22, x0 + bw * frac, H - 10], fill=AMBER if unc > gate_thr else TEAL)
        d.line([x0 + bw / 2, H - 25, x0 + bw / 2, H - 7], fill=INK, width=2)
    if info["faults"]:
        d.text((W - 10 - 9 * len(", ".join(info["faults"])), 40), ", ".join(info["faults"]), fill=(194, 59, 38), font=fnt)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="teleop", choices=["teleop", "autonomous", "hitl"])
    ap.add_argument("--policy", help="policy checkpoint (required for autonomous/hitl)")
    ap.add_argument("--faults", default="none")
    ap.add_argument("--force-faults", nargs="*", default=[], help="e.g. slip fixture_shift ood_pose")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--camera", default="overview")
    ap.add_argument("--threshold", type=float, default=0.12)
    ap.add_argument("--out", default="cell.mp4")
    args = ap.parse_args()

    env = MachineTendingEnv(faults=args.faults, render_mode="rgb_array", camera=args.camera, width=960, height=720)
    policy = EnsemblePolicy.load(args.policy) if args.policy else None
    if args.mode != "teleop" and policy is None:
        ap.error("--policy is required for autonomous/hitl")
    fnt = font(15)
    writer = imageio.get_writer(args.out, fps=env.metadata["render_fps"], macro_block_size=8)

    def cb(env, info, who, unc):
        writer.append_data(overlay(env.render(), info, who, unc, args.threshold if args.mode == "hitl" else 0, fnt))

    for ep in range(args.episodes):
        log = run_episode(env, args.mode, policy=policy, operator=ScriptedOperator(),
                          gate=Gate(threshold=args.threshold), seed=args.seed + ep, frame_cb=cb,
                          options={"force_faults": args.force_faults})
        print(f"episode {ep}: success={log.success} asks={[(round(t, 1), r) for t, r, _ in log.asks]} "
              f"faults={log.faults}")
    writer.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
