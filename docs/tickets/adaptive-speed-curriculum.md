# Ticket: performance-gated (adaptive) speed-ceiling curriculum

**Status:** not started · **Filed:** 2026-08-30 · **Priority:** medium

## Problem

Every velocity-command curriculum in this repo is **step-based** — fixed ranges
at fixed iteration counts:

- `microduck_velocity_straightleg_env_cfg.py` → `speed_ceiling`, via
  `microduck_mdp.velocity_command_ranges_curriculum`
- mjlab's own `velocity_env_cfg.py` → `command_vel`, via `mdp.commands_vel`

The only performance-gated curriculum anywhere in the stack is
`terrain_levels_vel`, which promotes/demotes terrain difficulty based on how far
the robot actually walked. Nothing does that for command ranges.

Step-based ranges have three concrete failure modes, all observed:

1. **They encode a guess about when a skill will exist.** AGENTS.md: phase-align
   every stage with what the policy has actually learned. A fixed step number
   cannot do that.
2. **They can command the unreachable.** If the ceiling outruns the robot, the
   policy trains against *inescapable* error — it cannot fix the gap, so the
   gradient just pushes toward maximum effort everywhere. The velocity env's own
   header records this: "a ramp to lin ±0.4 / ang ±2.0 outpaced the robot's
   capability and tracked a post-iter-1000 reward/episode-length decline."
3. **They are resume-fragile.** Steps are ABSOLUTE and `common_step_counter` is
   restored from the checkpoint (mjlab/rl/runner.py). On 2026-08-30 a resume
   from iteration 1750 landed past both stages of a 0.4 → 0.5 → 0.6 ramp, so the
   ceiling jumped straight to 0.6 instead of ramping. The gradual introduction
   that was deliberately designed simply did not happen.

## Proposal

A curriculum that raises the `lin_vel_x` ceiling only when the policy is
actually tracking the current one.

```
if mean(Metrics/twist/error_vel_xy) < PROMOTE_THRESHOLD for N consecutive
   iterations and ceiling < MAX:
       ceiling += STEP
```

Design questions to settle when implementing:

- **One-way or two-way?** Allowing the ceiling to drop when error climbs is more
  robust but can oscillate at the frontier. One-way ratchet is simpler and
  matches how `speed_ceiling` reads today. Suggest starting one-way with a
  hysteresis band, and only adding demotion if it proves necessary.
- **What metric?** `error_vel_xy` is the obvious candidate but it mixes x and y.
  Since only `lin_vel_x` widens, a forward-only error signal would be cleaner.
- **Threshold value** should be derived from measurement, not guessed — see the
  data below for what "good tracking" currently looks like.
- Must be resume-safe by construction: key off measured performance, never off
  `common_step_counter`. That is the main point of the ticket.

## Supporting measurement (2026-08-30, checkpoint iter 4750)

The current policy tracks its whole range well, so an adaptive curriculum would
have promoted freely here — the value is in the runs where it would NOT:

| commanded (m/s) | achieved | % of command |
|---|---|---|
| 0.3 | 0.314 | 104.7% |
| 0.4 | 0.405 | 101.3% |
| 0.5 | 0.498 | 99.6% |
| 0.6 | 0.593 | 98.8% |

No saturation up to the 0.6 ceiling. Left-knee deviation stayed at 0.461° mean
at 0.3 m/s and 0.671° at 0.6 m/s, so the straight-leg constraint holds at speed.

## Explicitly rejected alternative: upsampling high-speed commands

Considered and not recommended as the primary mechanism. Upsampling changes the
*density* within a fixed range; it cannot prevent unreachable commands, which is
the actual risk. The curriculum changes the *range itself*, keeping every
command achievable by construction.

Note also that high speed is not especially rare under uniform sampling —
`P(vx > 0.5) ≈ 8.3%` at a 0.6 ceiling. (An earlier estimate of ~0.14% was for
the *corner* — fast AND straight AND not turning — which is a different and
narrower claim.) Unlike turn-in-place, which is a thin slice of the command box
AND a behaviourally distinct maneuver, high speed is a face of the box reached
by continuous variation of the same gait. It does not obviously need a bucket.

Upsample only if measurement shows tracking error is materially worse at high
commands than mid-range. As of iter 4750 it is not.

## Scope

- New curriculum function in `src/mjlab_microduck/tasks/mdp.py`, next to
  `velocity_command_ranges_curriculum` (do not delete that one — other envs use it).
- Wire into `microduck_velocity_straightleg_env_cfg.py` in place of the current
  step-based `speed_ceiling`.
- Cfg test in `tests/test_straightleg_cfg.py`: ceiling never exceeds the
  configured max, never widens `lin_vel_y` / `ang_vel_z`, and starts at the
  base range.
- ~30 lines plus the test. Config-only — does not require touching a live run.
