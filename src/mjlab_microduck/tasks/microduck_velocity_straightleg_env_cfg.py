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
LOCK_WEIGHT = 2.0
L1_WEIGHT_STAGES = (3.0, 6.0, 9.0)  # iters 0 / 500 / 1000
KNEE_VEL_WEIGHT = -0.02
GATED_TRACK_WEIGHT = 1.0
# The base tracking term keeps the remaining mass (velocity env ships 2.0).
UNGATED_TRACK_WEIGHT = 1.0

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
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": L1_WEIGHT_STAGES[2]},
            ],
        },
    )

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
