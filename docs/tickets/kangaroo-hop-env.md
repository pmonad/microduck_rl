# Ticket: kangaroo-hop locomotion env

**Status:** not started · **Filed:** 2026-08-30 · **For:** handover to a fresh session

## Goal

A new task where the Microduck locomotes by **two-footed synchronised hopping**
(kangaroo/pogo style) instead of an alternating walking gait, while still
tracking a commanded twist. Both feet leave and land **together**, with a real
flight phase.

This is a sibling of the existing `Mjlab-VelocityStraightLeg-Flat-MicroDuck`
task (branch `feat/straightleg-env`), which constrained a walking gait. Same
idea — inherit the velocity recipe, change only what defines the maneuver.

## Read first

- `AGENTS.md` — non-negotiable. Especially "Reward design", "Building a new env",
  and the 61D obs invariant.
- `src/mjlab_microduck/tasks/microduck_velocity_straightleg_env_cfg.py` — the
  closest worked example of "velocity recipe + one constraint", including a
  docstring that explains *why* each layer exists.
- `src/mjlab_microduck/tasks/microduck_velocity_swizzle_env_cfg.py` — the
  cleanest example of "drop the terms that fight the new gait, add the terms
  that define it".

## STEP 0 — feasibility, before writing any reward

**Do this first.** AGENTS.md step 2: verify physics assumptions in sim before
training. It is not established that an ~800 g, 25 cm robot on XL330 servos can
hop at all — the servos may not have the burst power to leave the ground.

Measure it: script a ballistic crouch-and-extend (drive both legs from a deep
crouch to full extension at max ctrl) and record peak trunk `z` and whether
both `left_foot_collision` / `right_foot_collision` lose contact simultaneously.

- If peak flight height is **> ~15 mm** with clean two-foot takeoff → proceed.
- If it barely leaves the ground → **stop and report back before training.**
  The task may need to be redefined as "bounding/skipping with a brief
  double-flight" rather than true hopping. Do not burn a 4-hour run finding
  this out.

Record the measured numbers in this ticket.

## Design sketch

Build on `make_microduck_velocity_env_cfg` (do NOT start from mjlab's base
template — you would have to re-port the whole DR + obs-noise + NaN-guard stack).

**Remove** the terms that reward an alternating gait:
- `air_time` — rewards per-foot air time in a window, which is exactly the
  single-support alternation we are trying to replace.
- `gait_symmetry` / any anti-synchrony term, if present.

**Add** the terms that define hopping:
1. **Foot synchrony** — reward the two feet being in the *same* contact state.
   `leg_symmetry_reward` (mdp.py:4792) already exists for the swizzle env and
   may be reusable directly; check it before writing a new one.
2. **Flight phase** — reward both feet simultaneously off the ground, but see
   the jackpot warning below.
3. **Hop cadence / vertical oscillation** — reward trunk-z oscillation at a
   target frequency, or peak apex height per cycle. Prefer paying Δprogress
   over paying per-step for being airborne.
4. Keep `track_linear_velocity` / `track_angular_velocity` and the head-pose
   stack unchanged — this is still a commanded-velocity task.

## Pitfalls specific to this task

- **The flight-phase jackpot.** "Both feet off the ground" is trivially
  satisfied by *falling over*, by lying on its back, or by one giant leap
  followed by nothing. AGENTS.md: no jackpots, and never gate a positive reward
  on being in a bad state. Gate any airborne reward on trunk height AND upright
  orientation AND forward progress, and rate-limit it so a single huge leap
  cannot out-earn steady hopping.
- **Regularizers will block the skill.** `body_ang_vel` and `angular_momentum`
  are motion-blockers; a hop physically requires vertical momentum and pitch
  oscillation. Keep them LOW (AGENTS.md says this explicitly for dynamic
  tasks). The straight-leg env inherits `body_ang_vel=-0.05`,
  `angular_momentum=-0.02` — those were tuned for walking and may already be
  too high here.
- **Introduce `action_rate` AFTER discovery.** Hopping is a hard skill; an
  attempt-tax active during exploration makes "stand still" the argmax. The
  velocity recipe ramps action_rate from −0.1 to −1.0 by iter 1500 — consider
  holding it near 0 for longer.
- **Sign convention.** mjlab-base costs (≥ 0) take negative weights;
  microduck `*_penalty` / `*_l1` functions self-negate (≤ 0) and take POSITIVE
  weights. Verify on every run: every `Episode_Reward/<penalty>` must be ≤ 0.

- **THE STAND-STILL TRAP — expect this to bite harder here than it did for
  walking.** Added 2026-08-30 after measuring it on the straight-leg policy (see
  `low-speed-dead-zone.md`). `track_linear_velocity` is
  `exp(-error² / std²)` with `std² = 0.1`, so standing still against a commanded
  0.1 m/s scores `exp(-0.01/0.1)` = **0.905 — 90.5% of maximum tracking reward
  for doing nothing.** The straight-leg policy consequently refuses to move at
  all below ~0.3 m/s.

  A hop policy is *more* exposed to this, not less: hopping costs far more
  action-rate, torque and impact penalty than walking does, so the reward
  margin that must justify moving is larger, while the tracking payoff for
  standing still is identical. If you inherit the velocity recipe's tracking std
  unchanged, the most likely failure mode of this whole task is a robot that
  stands perfectly still and collects 90% of the tracking reward.

  Decide the tracking std deliberately before the first real run, and
  **sweep commanded speed from 0.1 to max** in evaluation — do not evaluate only
  at mid-range. The straight-leg audit measured 83%/100% tracking at 0.3-0.4 m/s
  and looked healthy for hours before a low-speed sweep exposed the dead zone.

## Things that differ from the straight-leg task

- **Symmetry mirror-loss can be ENABLED here.** Hopping is bilaterally
  symmetric, so `SYMMETRY_CFG` in `symmetry.py` is valid and should improve
  sample efficiency. (The straight-leg task had to force it off — it was
  asymmetric by construction.) Worth an A/B.
- Feet must lift *together*, so `foot_clearance` / `foot_swing_height` targets
  probably need revisiting rather than inheriting.

## Boilerplate checklist

- `ENABLE_*` toggles + tuned constants at the top of the cfg module.
- Factory `make_microduck_hop_env_cfg(play: bool, rough: bool)`.
- Register Flat / Rough in `tasks/__init__.py`, plus the `-Backlash-` variant
  in `_BACKLASH_TASKS` using `_BL_WALK` (mirror the base task's robot model).
- Own `RslRl...RunnerCfg` with a distinct `experiment_name` (e.g. `velocity_hop`).
- **Obs stays 61D.** Keep every command slot even if unused — zero-pad, never
  delete.
- Cfg tests in `tests/test_hop_cfg.py`: joint indices resolve on the real model,
  reward weights have the intended sign, gates open/close where expected, and
  building the env does not mutate an existing velocity cfg (the base template
  shares mutable objects — this bit us once).

## Commands

```bash
uv run list-envs
uv run train <TASK_ID> --env.scene.num-envs 64 --agent.max_iterations 5   # SMOKE FIRST, always
uv run train <TASK_ID> --env.scene.num-envs 8192
uv run scripts/export.py <TASK_ID> --checkpoint-file <ckpt> --onnx-file /tmp/hop.onnx --device cpu --num-envs 1
MUJOCO_GL=egl uv run scripts/infer_policy.py --walking /tmp/hop.onnx --new-cmd-obs \
  --lin-vel-x 0.3 --video /tmp/hop.mp4 --duration 15 --fast
```

Three tooling facts that cost real time to rediscover:

- `--new-cmd-obs` is **mandatory** for 61D policies. Without it `infer_policy.py`
  silently builds a 51D observation and renders convincing nonsense.
- Pass `--lin-vel-x` etc. explicitly, or the rollout runs at zero command and you
  will be watching the robot stand still.
- **The headless viewer's camera TRACKS `trunk_base` at `distance = 1.0`**
  (`HeadlessViewer.__init__`, `track_body="trunk_base"`). For a walking policy
  this merely hides translation. **For a hop it is much worse: a body-tracking
  camera cancels the vertical motion too, so a robot that is hopping correctly
  will look like it is standing still.** Do not judge hop height from these
  videos. Either add a fixed-camera / `--no-track` option first, or measure
  apex height from `trunk_z` telemetry and contact data. A healthy 0.302 m/s
  walking rollout was reported as "not moving" on 2026-08-30 for exactly this
  reason.

`--save-csv` **exists and works** — confirmed 2026-08-30, it dumps 77 columns ×
600 rows (joint positions among them) for a 12 s rollout. Use it for the flight-
phase and contact analysis rather than writing a bespoke rollout script.

## Free performance (measured 2026-08-30, applies to any flat env)

On flat terrain, `njmax=256` + `ls_parallel=False` gave **+30.5%** throughput
(37.0k → 48.3k steps/s on a GB10; +23% on an RTX 5090). `njmax` is safe because
an instrumented rollout (2048 envs, 400 steps, random actions so robots fall and
pile up contacts) measured **peak nefc = 59** against a default of 1500 — 4.3×
headroom, so it is a capacity change, not a physics change.

**Re-measure nefc for the hop env before copying this.** Hopping produces
different contact patterns (hard two-foot landings), and rough terrain was
excluded from the straight-leg version for the same reason.

Also measured and NOT worth taking: `jacobian dense/sparse` (−1%/−8%),
`cone elliptic` (−12%), `ccd-iterations 12` (±0%).

## Curricula and resume (learned the hard way, 2026-08-30)

Curriculum stage `step` numbers are **ABSOLUTE env steps, and they survive a
resume**: mjlab's runner (`mjlab/rl/runner.py`, "Restore common_step_counter to
preserve curricula state") saves the counter into the checkpoint and restores
it. `ManagerBasedRlEnv.__init__` does set it to 0, but the runner overwrites
that — reading only the env gives the wrong answer.

Consequence to design around: resuming from a checkpoint **past** a stage
boundary makes that stage fire instantly rather than ramp. A speed ramp
designed as 0.4 → 0.5 → 0.6 collapsed into an immediate jump to 0.6 this way.
If you resume a hop run and expect a gradual introduction of anything, place the
stages past the resume point.

See also `adaptive-speed-curriculum.md` — a performance-gated curriculum would
sidestep this entirely, and would suit hop-height/speed ramps well.

## Acceptance criteria

1. Smoke test passes (64 envs, 5 iters): builds, 61D obs, NaN-free, every reward
   term computes, ONNX exports.
2. Rollout video shows **both feet leaving the ground together** with a
   measurable flight phase — verified from contact data, not by eye.
3. Forward velocity tracks the command at a comparable fraction to the walking
   policy (straight-leg reached 83% of 0.4 m/s).
4. Only foot geoms contact the ground; no knee/trunk/head strikes on landing.
5. Report what rollouts actually show, including failure rate — not "it works".

## Open questions for whoever picks this up

- True flight phase, or is a brief double-support-free "skip" acceptable if
  STEP 0 shows the servos cannot clear the ground?
- Target hop frequency — should it be commanded (a new obs slot would break the
  61D contract, so more likely fixed or derived from commanded speed)?
- Should hop height scale with commanded forward speed, like a real kangaroo?
