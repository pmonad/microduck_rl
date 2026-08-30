"""Microduck STRAIGHT-LEFT-LEG walking environment.

Walk on velocity command while the LEFT knee stays locked straight — a
peg-leg / knee-immobilizer gait. Everything that makes the walking recipe
transfer (robot, 61D obs, commands, full DR stack, obs noise + delays, NaN
guards, head-pose tracking, curricula) is inherited from
``make_microduck_velocity_env_cfg``; only the constraint is added.

What "straight" means here: ``HOME_FRAME`` sets left_knee = -0.0049 rad, so
the HOME pose ALREADY is the straight-leg pose. The target is therefore
``default_joint_pos`` and the task is "never flex the left knee", not "reach
a new pose". Hip pitch/roll/yaw and the ankle stay free — a rigid *thigh+shank*
line is what a straight leg is, and the robot needs hip circumduction and
ankle push-off to swing a stiff leg at all (that's how a human walks in a knee
brace). Constraining them too would make the task "hop on one leg" instead.

Design notes:

- The constraint is ON FROM STEP 0 at moderate strength, not curriculum'd in
  late. The usual AGENTS.md rule ("don't introduce taxes before the skill
  exists") does not apply: the robot *starts* satisfying this constraint (HOME
  knee = straight), so it is not a tax on a hard skill — it is a boundary on
  the gait being discovered. Ramping it in later would mean first learning a
  knee-bending gait and then unlearning it, which is strictly harder.
- Four layers, deliberately: a multiplicative gate on tracking (kills the
  "bend a bit, keep most of the speed reward" compromise basin), a sharp
  Gaussian lock (peak at zero error), an L1 (constant gradient when far), and
  a knee-velocity penalty (stops the knee flapping through straight and
  averaging out).
- Tracking reward MASS is conserved, not duplicated: the base
  ``track_linear_velocity`` weight drops 2.0 → 1.0 and the gated composite
  takes the other 1.0, so the task stack does not inflate relative to the
  inherited regularizers (AGENTS.md: "compare reward mass, not weights").
  The residual ungated 1.0 keeps a bootstrap gradient for walking at all.
- Symmetry mirror-loss MUST stay off: this task is asymmetric by construction
  (left ≠ right), so the mirror loss would actively fight it. Enforced in
  ``MicroduckStraightLegRlCfg`` below rather than inherited on trust.

Not touched, and worth watching on the first real run: ``foot_clearance`` /
``foot_swing_height`` still ask BOTH feet for 2 cm of lift. A locked left knee
has to earn that clearance from hip flexion + circumduction + right-side
stance shortening. If the left foot ends up dragging, lower those targets (or
the straightness weights) rather than assuming the constraint is at fault.
"""

import dataclasses
import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import rewards as base_rewards
from mjlab.managers import CurriculumTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    NUM_STEPS_PER_ENV,
    make_microduck_velocity_env_cfg,
)

# Which leg is immobilized. Index is into the canonical 14-servo view (see
# AGENTS.md "Joint layout"); the name pattern excludes passive_* so it also
# resolves correctly on the backlash model.
STRAIGHT_KNEE_IDX = microduck_mdp.LEFT_KNEE_IDX  # 3 = left_knee
STRAIGHT_KNEE_PATTERN = r"^(?!passive_).*left_knee.*"

# Gaussian std for the sharp lock reward: ≈ the error we still care about
# (0.05 rad ≈ 2.9°), not the max error — a looser std has no gradient where it
# matters, and the L1 term below already covers the far field.
LOCK_STD = 0.05
# Gate std inside the tracking composite: WIDE on purpose (0.20 rad ≈ 11.5°) so
# a policy that currently bends the knee still scores visibly and can see which
# way is up. The sharp peak is LOCK_STD's job.
GATE_STD = 0.20

# Weights. Note the two sign conventions (AGENTS.md): pose_l1_penalty is a
# self-negating microduck penalty (returns ≤ 0) → POSITIVE weight; joint_vel_l2
# is an mjlab-base cost (returns ≥ 0) → NEGATIVE weight. Both must log
# Episode_Reward ≤ 0.
#
# Run-1 audit (checkpoint 1250/1750, 8192 envs) drove the values below:
#   - The constraint WORKS: left knee held 0.44° mean / 2.34° max error while
#     the right knee (control) swept 38° peak-to-peak — ~10x separation. The
#     gate ratio reached 0.996 and the robot walked at 83% of commanded 0.4 m/s
#     on foot contacts only, with near-symmetric foot lift (15.1 vs 16.1 mm).
#   - So the lock was OVER-PAID: straight_leg_lock took 20% of the positive
#     reward mass with only 0.129 headroom left, buying a constraint already
#     satisfied 10x over. Halved 2.0 → 1.0 and the mass moved to tracking.
LOCK_WEIGHT = 1.0  # was 2.0 — see audit note above
# Run-1 dropped the third L1 stage (6→9 at iter 1000): it improved knee error
# by only 4% (0.63° → 0.60°) while air_time fell 9% and peak foot height 10%,
# NEITHER recovering over the following 380 iterations — AGENTS.md's "steps
# DOWN at a curriculum boundary means the pacing is wrong" signal. Caveat: the
# action_rate ramp stepped at the same iteration so attribution is not clean,
# but the identical action_rate step at iter 750 caused only a transient.
L1_WEIGHT_STAGES = (3.0, 6.0)  # was (3.0, 6.0, 9.0)
KNEE_VEL_WEIGHT = -0.02
# Tracking mass raised 2.0 → 3.0 total. The audit found velocity tracking, not
# straightness, is the weak axis: forward reached 83% of command, backward 58%,
# lateral only 24%, and error_vel_xy was improving just 6% per 400 iters. This
# is where the reward mass freed from LOCK_WEIGHT goes.
GATED_TRACK_WEIGHT = 1.5
UNGATED_TRACK_WEIGHT = 1.5

# ── Speed push ────────────────────────────────────────────────────────────────
# "Go as fast as possible", implemented WITHOUT breaking the velocity-command
# contract: this stays a tracking task (the runtime steers the robot by sending
# a twist command, so a policy that always sprints would be undeployable).
# Speed is pushed by raising the forward command CEILING on a curriculum, plus
# the tracking-mass increase above.
#
# Staged LATE and deliberately: the audit measured the robot reaching only 83%
# of its CURRENT 0.4 m/s ceiling, so raising the ceiling before it can hit the
# existing one is exactly the documented failure in the velocity env's header
# ("a ramp to lin ±0.4 / ang ±2.0 outpaced the robot's capability and tracked a
# post-iter-1000 reward/episode-length decline").
#
# NOTE ON STEP NUMBERS: these are ABSOLUTE. mjlab's runner (mjlab/rl/runner.py,
# "Restore common_step_counter to preserve curricula state") saves
# env.common_step_counter into the checkpoint and restores it on --agent.resume,
# so curricula pick up exactly where they left off. ManagerBasedRlEnv.__init__
# does set the counter to 0, but the runner overwrites it — do not be misled by
# reading only the env.
#
# CONSEQUENCE, learned the hard way: run 2 resumed from iteration 1750, which is
# already past both stages below, so the ceiling jumped 0.4 → 0.6 immediately
# instead of ramping. It survived that (error_vel_xy still improved ~6%), but
# any FUTURE stage added here must be placed past the resume point to actually
# ramp.
SPEED_CMD_STAGES = (
    (0, 0.4),
    (800, 0.5),
    (1600, 0.6),
)
# Only lin_vel_x widens. lin_vel_y is left at its ±0.3 (lateral is the WEAKEST
# axis at 24% of command — widening it would add error, not speed) and
# ang_vel_z at ±1.0 (yaw already tracks at 107%, it needs no help).

# Pose-reward std for the locked knee. The velocity recipe uses 0.15 (standing)
# / 0.4 (walking) for BOTH knees; leaving that in place would let the pose
# reward saturate on a visibly bent left knee and quietly undercut the lock.
POSE_STD_LOCKED_KNEE_STANDING = 0.04
POSE_STD_LOCKED_KNEE_WALKING = 0.06


def _tighten_locked_knee_pose_std(cfg: ManagerBasedRlEnvCfg) -> None:
    """Split the shared ``.*knee.*`` std entry into per-side entries.

    ``resolve_matching_names_values`` raises on a joint matching two keys, so a
    ``.*left_knee.*`` override cannot simply be added alongside ``.*knee.*`` —
    the generic key must become ``.*right_knee.*`` at the same time.
    """
    for std_key, locked_std in (
        ("std_standing", POSE_STD_LOCKED_KNEE_STANDING),
        ("std_walking", POSE_STD_LOCKED_KNEE_WALKING),
        ("std_running", POSE_STD_LOCKED_KNEE_WALKING),
    ):
        std_dict = dict(cfg.rewards["pose"].params[std_key])
        generic = std_dict.pop(r".*knee.*")
        std_dict[r".*left_knee.*"] = locked_std
        std_dict[r".*right_knee.*"] = generic
        cfg.rewards["pose"].params[std_key] = std_dict


def make_microduck_velocity_straightleg_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Velocity walking env with the left knee constrained straight."""
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    knee_cfg = SceneEntityCfg("robot", joint_names=(STRAIGHT_KNEE_PATTERN,))

    _tighten_locked_knee_pose_std(cfg)

    # 1) Multiplicative gate: speed is only worth having with the leg locked.
    #    Half the tracking mass moves here; the base term keeps the other half
    #    so walking still has an ungated bootstrap gradient.
    cfg.rewards["track_linear_velocity"].weight = UNGATED_TRACK_WEIGHT
    cfg.rewards["straight_leg_tracking"] = RewardTermCfg(
        func=microduck_mdp.straight_leg_velocity_tracking,
        weight=GATED_TRACK_WEIGHT,
        params={
            "command_name": "twist",
            "std": math.sqrt(0.1),  # same as track_linear_velocity
            "straight_std": GATE_STD,
            "joint_indices": (STRAIGHT_KNEE_IDX,),
        },
    )

    # 2) Sharp Gaussian peak at knee = HOME (straight).
    cfg.rewards["straight_leg_lock"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=LOCK_WEIGHT,
        params={"std": LOCK_STD, "joint_indices": [STRAIGHT_KNEE_IDX]},
    )

    # 3) L1 bootstrap — constant gradient even at large knee flexion, where the
    #    Gaussians above are flat. Self-negating penalty → POSITIVE weight.
    cfg.rewards["straight_leg_l1"] = RewardTermCfg(
        func=microduck_mdp.pose_l1_penalty,
        weight=L1_WEIGHT_STAGES[0],
        params={"joint_indices": [STRAIGHT_KNEE_IDX]},
    )

    # 4) Knee-velocity damper: without it the knee can flap through straight and
    #    still average well on the position terms. mjlab-base cost (≥ 0) →
    #    NEGATIVE weight. Kept light — it is a smoothness term, not a
    #    motion-blocker on the joints that must move.
    cfg.rewards["straight_leg_knee_vel"] = RewardTermCfg(
        func=base_rewards.joint_vel_l2,
        weight=KNEE_VEL_WEIGHT,
        params={"asset_cfg": knee_cfg},
    )

    # L1 ramp: the gate + lock shape the gait from step 0; the L1 hardens as the
    # gait consolidates so late residual flexion gets progressively less
    # affordable. Stages are phase-aligned with the inherited action_rate ramp
    # (which reaches full strength at iter 1500).
    cfg.curriculum["straight_leg_l1_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "straight_leg_l1",
            "weight_stages": [
                {"step": 0, "weight": L1_WEIGHT_STAGES[0]},
                {"step": 500 * NUM_STEPS_PER_ENV, "weight": L1_WEIGHT_STAGES[1]},
            ],
        },
    )

    # Forward-speed ceiling curriculum (see the SPEED_CMD_STAGES block above).
    # update_lin_vel_y / update_ang_vel_z are False so ONLY lin_vel_x widens —
    # the helper would otherwise slave lin_vel_y to the same range, which would
    # widen the weakest-tracking axis.
    cfg.curriculum["speed_ceiling"] = CurriculumTermCfg(
        func=microduck_mdp.velocity_command_ranges_curriculum,
        params={
            "command_name": "twist",
            "update_lin_vel_y": False,
            "update_ang_vel_z": False,
            "velocity_stages": [
                {
                    "step": it * NUM_STEPS_PER_ENV,
                    "lin_vel_range": v,
                    "ang_vel_range": 1.0,  # unused (update_ang_vel_z=False)
                }
                for it, v in SPEED_CMD_STAGES
            ],
        },
    )

    # ── Measured performance settings (flat only) ─────────────────────────────
    # njmax caps the constraint-array capacity. mjlab defaults it to 1500 here,
    # but an instrumented rollout (2048 envs, 400 steps, RANDOM actions so the
    # robots fall and pile up multi-limb ground contacts — the worst case for
    # constraint count) measured a peak nefc of just 59. 256 keeps 4.3x headroom
    # over that peak, so no constraint can ever be dropped: this is a CAPACITY
    # change, not a physics change. Measured +23.7% throughput on its own.
    # ls_parallel=False measured a further +9.1% (combined: +30.5%, 37.0k →
    # 48.3k steps/s at 4096 envs on a GB10). ls_parallel only changes how the
    # Newton solver's line search is parallelised, not the problem being solved.
    # Rough terrain is deliberately EXCLUDED: the nefc measurement was taken on
    # a plane, and box terrain adds contacts, so the headroom is unproven there.
    if not rough:
        cfg.sim.njmax = 256
        cfg.sim.ls_parallel = False

    return cfg


# Same PPO hyperparameters as the velocity task, new experiment/run name.
# symmetry_cfg is forced to None: the mirror loss asserts left/right
# interchangeability, which is exactly what this task breaks.
MicroduckStraightLegRlCfg = dataclasses.replace(
    MicroduckRlCfg,
    algorithm=dataclasses.replace(MicroduckRlCfg.algorithm, symmetry_cfg=None),
    experiment_name="velocity_straightleg",
    run_name="velocity_straightleg",
)
