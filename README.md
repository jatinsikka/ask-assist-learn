# Ask, Assist, Learn

Human-in-the-loop learning for long-horizon robot manipulation in manufacturing, on robot arms and humanoids.

Live site: https://jatinsikka.github.io/ask-assist-learn/

## The idea

A robot runs a standard operating procedure (SOP). When it is unsure, it asks for help; an operator assists in VR; the correction is logged; the skill is retrained. The question is how to make each step reliable with the least operator time.

The page covers:

- Why reliability limits deployment
- The run → ask → assist → learn loop
- Three autonomy layers plus a human layer
- Models and training: per-step visuomotor policies (Diffusion Policy or ACT), learned help requests, HG-DAgger / Sirius-style learning from corrections, and RL fine-tuning in Isaac Lab, with a VLA comparison (π0.5 via openpi, GR00T N1)
- Three research questions, related work, and a plan that starts with robot arms

## Simulation

`sim/` is a toy MuJoCo version of the loop: an arm tends a CNC machine, asks for help when its policy ensemble is unsure, and retrains on the operator's corrections. See [sim/README.md](sim/README.md).

## Files

- `index.html`: the whole site in one file (overview, supervisor console, 3D AR operator view)
- `.nojekyll`: serve the files as-is, without Jekyll
- `sim/`: the MuJoCo cell, policies and experiment scripts

The 3D view loads three.js r128 from cdnjs, and fonts load from Google Fonts. The console and 3D simulation are scripted for illustration, not physics.

Hosted with GitHub Pages from `main` / root.
