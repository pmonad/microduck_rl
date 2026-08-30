# Ticket: low-speed dead zone — policy stands still below ~0.3 m/s

**Status:** not started · **Filed:** 2026-08-30 · **Priority:** high (affects deployment)

## Symptom

Measured on `Mjlab-VelocityStraightLeg-Flat-MicroDuck`, checkpoint iter 4750,
12 s rollouts on the plain scene:

| commanded (m/s) | achieved | % of command | behaviour |
|---|---|---|---|
| **0.1** | **~0.00** | **0%** | never steps, trunk_z pinned flat |
| **0.2** | **~0.00** | **0%** | never steps, trunk_z pinned flat |
| 0.3 | 0.314 | 104.7% | walks |
| 0.4 | 0.405 | 101.3% | walks |
| 0.5 | 0.498 | 99.6% | walks |
| 0.6 | 0.593 | 98.8% | walks |

The robot does not fall or wobble — it stands perfectly still, with no gait
oscillation in trunk_z at all. This is a clean dead zone, not instability.

This matters for deployment: slow approach speeds are a normal command, and the
runtime would send 0.1–0.2 m/s expecting a slow walk, not a freeze.

## Root cause (arithmetic, not a mystery)

`track_linear_velocity` = `exp(-lin_vel_error / std**2)` with
`std = sqrt(0.1)`, i.e. `std**2 = 0.1`, where `lin_vel_error` is the SQUARED
velocity error.

Standing still against a commanded speed `v` therefore scores:

| command v | error² | reward for standing still |
|---|---|---|
| 0.1 | 0.01 | `exp(-0.10)` = **0.905** |
| 0.2 | 0.04 | `exp(-0.40)` = **0.670** |
| 0.3 | 0.09 | `exp(-0.90)` = 0.407 |
| 0.6 | 0.36 | `exp(-3.60)` = 0.027 |

At 0.1 m/s the policy collects **90.5% of the maximum tracking reward for doing
nothing**. Walking to earn the remaining 9.5% costs `action_rate_l2` (79.8% of
all penalty mass in this run) plus `joint_torque_rate_l2`. Standing still is
simply the better trade, and PPO found it.

This is AGENTS.md's documented failure exactly: *"Tracking Gaussian std: ≈ the
error you still care about, not the max error — too loose has no gradient at
small errors."* A std of 0.316 m/s is far too loose to distinguish "walking at
0.1" from "stationary".

Contributing factor worth checking: `rel_standing_envs` ramps to 0.25, so 25% of
envs are explicitly commanded to stand. Combined with the above, the policy may
be generalising "small command ⇒ stand" from the zero-command bucket.

## Candidate fixes (not yet evaluated — pick with measurement)

1. **Tighten the tracking std.** Directly attacks the cause. Risk: a tighter std
   raises the tax on the whole range, and AGENTS.md warns that an unescapable
   tight tracking std once made a policy stand still entirely (the head-tracking
   incident). Must check the error is escapable before tightening.
2. **Command-scaled std** — `std = max(std_floor, k * |cmd|)` — so tolerance
   shrinks with the command instead of being absolute. Keeps large-command
   behaviour unchanged while making small commands discriminative. Probably the
   cleanest, but needs a new mdp function.
3. **A stepping requirement when commanded non-zero**, e.g. reuse
   `no_stepping_penalty` (already in mdp.py) gated on `|cmd| > threshold`.
   Careful: must NOT fire in the zero-command idle state, which is trained
   deliberately.
4. **Verify the standing bucket isn't bleeding.** Check whether
   `rel_standing_envs` envs are being sampled with small-but-nonzero commands.

Whatever is chosen, verify with the same 0.1–0.6 sweep before and after.

## Reproduce

```bash
cd /home/pmonad/git/pollen-robotics/microduck_rl   # branch feat/straightleg-env
uv run scripts/export.py Mjlab-VelocityStraightLeg-Flat-MicroDuck \
  --checkpoint-file <ckpt> --onnx-file /tmp/p.onnx --device cpu --num-envs 1
for S in 0.1 0.2 0.3 0.4 0.5 0.6; do
  MUJOCO_GL=egl uv run scripts/infer_policy.py --walking /tmp/p.onnx --new-cmd-obs \
    --lin-vel-x $S --video /tmp/speed_$S.mp4 --duration 12 --fast
done
```

`--new-cmd-obs` is mandatory (61D policy; the default 51D path silently feeds a
wrong-sized obs vector). `--lin-vel-x` is mandatory or the command is zero and
everything looks like a dead zone.

## Related tooling annoyance (separate, low priority)

`scripts/infer_policy.py`'s headless viewer tracks `trunk_base` at
`distance = 1.0` (see `HeadlessViewer.__init__`, `track_body="trunk_base"`), so
on a featureless plane a *correctly walking* robot looks like it is walking in
place — only the checkerboard shift betrays motion. This cost real time today:
a healthy 0.302 m/s rollout was reported as "not moving".

Consider a `--no-track` / fixed-camera flag, or drawing a world-frame marker, so
review videos show translation unambiguously. Note this lives in an
**uncommitted local patch** to `infer_policy.py` on the spark box, not in git.
