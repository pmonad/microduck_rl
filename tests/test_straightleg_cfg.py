"""Cfg invariants for the straight-left-leg walking env.

CPU-only: builds the cfg and checks joint resolution, reward signs, and the
things that would silently break the constraint.
"""

import math
import xml.etree.ElementTree as ET

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_WALK_ROBOT_CFG,
    MICRODUCK_WALK_XML,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_straightleg_env_cfg import (
    STRAIGHT_KNEE_IDX,
    make_microduck_velocity_straightleg_env_cfg,
    MicroduckStraightLegRlCfg,
)

_STRAIGHT_TERMS = (
    "straight_leg_tracking",
    "straight_leg_lock",
    "straight_leg_l1",
    "straight_leg_knee_vel",
)


def test_left_knee_index_matches_the_walk_model_joint_order():
    # The reward terms address the knee by INDEX into the servo view; if the
    # model's joint order ever moves, they would silently lock the wrong joint.
    # Document order of the named chosen_actuator joints IS the servo order
    # (the freejoint and the unnamed <default> joints are excluded by the
    # name+class filter).
    root = ET.parse(MICRODUCK_WALK_XML).getroot()
    servo_joints = [
        j.get("name")
        for j in root.iter("joint")
        if j.get("class") == "chosen_actuator" and j.get("name")
    ]
    assert len(servo_joints) == 14
    assert servo_joints[STRAIGHT_KNEE_IDX] == "left_knee"
    assert STRAIGHT_KNEE_IDX == microduck_mdp.LEFT_KNEE_IDX


def test_home_pose_is_the_straight_leg_pose():
    # The rewards target default_joint_pos, which is only "straight" because
    # HOME parks the knee at ~0. A future HOME with a bent knee would turn every
    # straightness term into a lock on the WRONG angle without failing loudly.
    home = MICRODUCK_WALK_ROBOT_CFG.init_state.joint_pos
    assert abs(home[r".*left_knee.*"]) < math.radians(1.0)


def test_all_straight_leg_terms_are_present():
    cfg = make_microduck_velocity_straightleg_env_cfg()
    for name in _STRAIGHT_TERMS:
        assert name in cfg.rewards, name


def test_reward_signs_follow_the_two_conventions():
    cfg = make_microduck_velocity_straightleg_env_cfg()
    # Self-negating microduck penalty (returns <= 0) -> POSITIVE weight.
    assert cfg.rewards["straight_leg_l1"].weight > 0.0
    assert cfg.rewards["straight_leg_l1"].func is microduck_mdp.pose_l1_penalty
    # mjlab-base cost (returns >= 0) -> NEGATIVE weight.
    assert cfg.rewards["straight_leg_knee_vel"].weight < 0.0
    # Positive objectives.
    assert cfg.rewards["straight_leg_lock"].weight > 0.0
    assert cfg.rewards["straight_leg_tracking"].weight > 0.0


def test_tracking_mass_is_raised_deliberately():
    # Originally the gated composite merely took half of track_linear_velocity's
    # mass. After the run-1 audit the total was raised on purpose (velocity
    # tracking was the measured weak axis), so assert the new intent.
    base = make_microduck_velocity_straightleg_env_cfg()
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
        make_microduck_velocity_env_cfg,
    )

    vel = make_microduck_velocity_env_cfg()
    total = (
        base.rewards["track_linear_velocity"].weight
        + base.rewards["straight_leg_tracking"].weight
    )
    # Run-1 audit moved mass from the lock INTO tracking, so the straightleg
    # stack now carries MORE tracking mass than the velocity env it derives from.
    assert total > vel.rewards["track_linear_velocity"].weight


def test_terms_address_only_the_left_knee():
    cfg = make_microduck_velocity_straightleg_env_cfg()
    for name in ("straight_leg_lock", "straight_leg_l1"):
        assert cfg.rewards[name].params["joint_indices"] == [STRAIGHT_KNEE_IDX]
    assert cfg.rewards["straight_leg_tracking"].params["joint_indices"] == (
        STRAIGHT_KNEE_IDX,
    )
    knee_names = cfg.rewards["straight_leg_knee_vel"].params["asset_cfg"].joint_names
    assert knee_names == (r"^(?!passive_).*left_knee.*",)


def test_gate_std_is_wider_than_the_sharp_lock_std():
    # A gate as tight as the lock scores ~0 for a currently-bending policy and
    # its gradient becomes invisible.
    cfg = make_microduck_velocity_straightleg_env_cfg()
    gate = cfg.rewards["straight_leg_tracking"].params["straight_std"]
    lock = cfg.rewards["straight_leg_lock"].params["std"]
    assert gate > lock


def test_pose_reward_knee_std_is_split_per_side_and_tighter_on_the_left():
    # The shared ".*knee.*" entry would let the pose reward saturate on a bent
    # left knee. Splitting is also mandatory: resolve_matching_names_values
    # raises when a joint matches two keys.
    cfg = make_microduck_velocity_straightleg_env_cfg()
    for key in ("std_standing", "std_walking", "std_running"):
        stds = cfg.rewards["pose"].params[key]
        assert r".*knee.*" not in stds
        assert stds[r".*left_knee.*"] < stds[r".*right_knee.*"]


def test_speed_ceiling_curriculum_only_widens_forward():
    # "Go fast" must not widen lin_vel_y: lateral is the weakest-tracking axis
    # (24% of command in the run-1 audit), so widening it adds error, not speed.
    cfg = make_microduck_velocity_straightleg_env_cfg()
    p = cfg.curriculum["speed_ceiling"].params
    assert p["update_lin_vel_y"] is False
    assert p["update_ang_vel_z"] is False
    speeds = [s["lin_vel_range"] for s in p["velocity_stages"]]
    assert speeds == sorted(speeds)
    # Stage 0 must not raise the ceiling before the policy re-settles on resume.
    assert speeds[0] == cfg.commands["twist"].ranges.lin_vel_x[1]


def test_perf_settings_applied_on_flat_but_not_rough():
    # njmax=256 is justified by a measured peak nefc of 59 on a PLANE. Box
    # terrain adds contacts, so the headroom is unproven there and the
    # reduction must not leak into the rough variant.
    flat = make_microduck_velocity_straightleg_env_cfg()
    rough = make_microduck_velocity_straightleg_env_cfg(rough=True)
    assert flat.sim.njmax == 256
    assert flat.sim.ls_parallel is False
    assert rough.sim.njmax != 256
    # Whatever it is, it must stay far above the measured peak of 59.
    assert flat.sim.njmax > 59 * 4


def test_lock_no_longer_dominates_tracking_mass():
    # Run-1 audit: the lock took 20% of positive mass for a constraint already
    # satisfied to 0.44 deg, while velocity tracking was the weak axis.
    cfg = make_microduck_velocity_straightleg_env_cfg()
    track = (cfg.rewards["track_linear_velocity"].weight
             + cfg.rewards["straight_leg_tracking"].weight)
    assert track > cfg.rewards["straight_leg_lock"].weight


def test_l1_curriculum_only_hardens():
    cfg = make_microduck_velocity_straightleg_env_cfg()
    stages = cfg.curriculum["straight_leg_l1_weight"].params["weight_stages"]
    assert stages[0]["weight"] == cfg.rewards["straight_leg_l1"].weight
    weights = [s["weight"] for s in stages]
    assert weights == sorted(weights)
    assert all(w > 0.0 for w in weights)


def test_symmetry_is_disabled_for_this_asymmetric_task():
    # A left/right mirror loss directly contradicts a left-only constraint.
    assert MicroduckStraightLegRlCfg.algorithm.symmetry_cfg is None
    assert MicroduckStraightLegRlCfg.experiment_name == "velocity_straightleg"


def test_obs_layout_stays_61d_across_both_groups():
    cfg = make_microduck_velocity_straightleg_env_cfg()
    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        assert "head_command" in terms
        assert "body_command" in terms
        # Command slot order is part of the shared runtime obs contract.
        assert list(terms).index("head_command") < list(terms).index("body_command")


def test_building_this_env_does_not_mutate_an_existing_velocity_cfg():
    # This env LOWERS track_linear_velocity's weight. make_velocity_env_cfg()
    # is documented to return shared mutable references, and the straightleg
    # tasks are registered AFTER the velocity ones — so if the reward term cfg
    # were shared, the main walking task would silently end up training at half
    # tracking weight. Build order here mirrors tasks/__init__.py.
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
        make_microduck_velocity_env_cfg,
    )

    vel = make_microduck_velocity_env_cfg()
    before = vel.rewards["track_linear_velocity"].weight
    make_microduck_velocity_straightleg_env_cfg()
    assert vel.rewards["track_linear_velocity"].weight == before
    assert "straight_leg_lock" not in vel.rewards
    assert r".*knee.*" in vel.rewards["pose"].params["std_walking"]
