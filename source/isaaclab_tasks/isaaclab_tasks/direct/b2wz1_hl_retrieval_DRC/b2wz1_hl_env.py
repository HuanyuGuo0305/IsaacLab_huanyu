# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import re
import torch

from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.utils.math as math_utils

from isaaclab_assets.robots.unitree import UNITREE_B2WZ1_CFG

from .modules.ll_policy_wrapper import LLPolicyWrapper
from .modules.command_adapter import build_keypoints_from_kp0_yaw_pitch_plb
from .modules.plb_frame import (
    transform_points_w_to_plb,
)

# Environment parameters
ENV_PARAMS = {
    # Frozen low-level policy
    "ll_task": "Isaac-Unitree-B2WZ1-PLB-WholeBody-v0",
    "ll_experiment_name": "unitree_b2wz1_plb_wholebody_loco_manip",
    "ll_load_run": "2026-07-17_14-28-55",
    "ll_load_checkpoint": "exported/policy.pt",

    # High-level timing
    # sim dt = 1 / 200
    # ll_decimation = 4 -> low-level at 50 Hz
    # ll_steps_per_hl_step = 5 -> high-level at 10 Hz
    "ll_steps_per_hl_step": 5,
    "ll_decimation": 4,

    # Low-level observation
    "ll_obs_history": 5,

    # PLB frame
    "ground_z": 0.0,

    # Low-level command limits
    "base_cmd_scale": [0.5, 0.5, 0.5],

    # Optional Stage-2 execution heuristics. Both are disabled by default so
    # the policy remains responsible for base stillness and gripper closure.
    "stage2_force_base_still_enabled": False,
    "stage2_force_gripper_close_enabled": True,

    "kp_dx": 0.30,
    "kp_dz": 0.30,

    # Neutral EE command used by the frozen low-level policy after reset.
    # Keep these values consistent with the original LL training configuration.
    "neutral_kp0": [0.55, 0.0, 1.12],
    "neutral_ee_yaw": 0.0,
    "neutral_ee_pitch": -0.06,
    # Roll is fixed and is not controlled by the high-level policy.
    "fixed_ee_roll": math.radians(90.0),

    # High-level observation history.
    "obs_history": 3,
    "critic_obs_history": 3,
    "obs_noise": True,
    "object_pos_detection_noise": 0.01,
    # Uniform actor noise U(-0.02, 0.02) m on each gripper-center component.
    "gripper_center_pos_noise": 0.02,
    # Per-frame Gaussian small-rotation noise applied jointly to the gripper
    # local +X/+Y axes in the actor observation. This preserves orthonormality.
    "gripper_orientation_noise_std": math.radians(1.0),
    "object_gripper_error_noise": 0.01,
    "object_target_error_noise": 0.01,

    # High-level EE command range
    "kp0_x_range": [0.40, 0.90],
    "kp0_y_range": [-0.30, 0.30],
    "kp0_z_range": [0.0, 1.15],

    "ee_yaw_range": [math.radians(-60.0), math.radians(60.0)],
    "ee_pitch_range": [math.radians(-70.0), math.radians(0.0)],

    # High-level arm actions are local deltas anchored at the current measured
    # gripperStator pose in PLB on every HL step:
    #   kp0_cmd = actual_ee_pos_plb + action[3:6] * kp0_delta_scale
    #   yaw_cmd = actual_ee_yaw_plb + action[6] * ee_yaw_delta_scale
    #   pitch_cmd = actual_ee_pitch_plb + action[7] * ee_pitch_delta_scale
    # The absolute command ranges above remain hard safety clamps.
    "kp0_delta_scale": [0.15, 0.15, 0.15],
    "ee_yaw_delta_scale": math.radians(10.0),
    "ee_pitch_delta_scale": math.radians(10.0),
    # Object sampling in the PLB frame used by the frozen low-level command.
    # Instead of sampling from a rectangular [x, y] box, sample from a
    # pregrasp-centered polar sector:
    #   r = U(r_min, r_max)
    #   theta = U(-theta_max, theta_max)
    #   object_x = x_center + r * cos(theta)
    #   object_y = y_center + r * sin(theta)
    # This makes early training focus on the base-ready/pregrasp sector.
    "object_sampling_center_x": 0.8,
    "object_sampling_center_y": 0.0,
    # Fixed sampling range.
    "object_sampling_r_range": [0.0, 2.0],
    "object_sampling_theta_max": math.radians(45.0),
    # Arm-only Beta-PPO sanity target. The cube center is placed at ground
    # contact height, so its PLB center is [0.8, 0.0, half_extent + epsilon].
    "sanity_object_xy_plb": [0.8, 0.0],
    "sanity_force_base_command_zero": False,
    "sanity_force_gripper_open": False,

    # Grasp object: a fixed 6 cm normal cube.
    "cube_side": 0.05,
    "object_half_extent": 0.025,
    # Spawn the cube slightly above the ground to avoid initial visual/solver jitter.
    # The cube pose is still read back from the real simulated RigidObject state;
    # no task/cache buffers are manually overwritten.
    "object_spawn_z_epsilon": 0.001,
    "ball_mass": 0.10,

    # Full mobile-manipulation grasp task.
    # Single canonical grasp-center reference expressed in the measured
    # gripperStator frame. Every grasp-center-dependent calculation uses
    # this same fixed point; it does not move with jointGripper closure.
    "gripper_center_offset_local": [0.075, 0.0, 0.01],

    # Stage-independent grasp alignment uses the SAME canonical center above.
    # Object center is expressed in gripperStator local coordinates and
    # compared directly with gripper_center_offset_local.
    "grasp_align_coarse_exp_std": 0.35,
    "grasp_align_fine_exp_std": 0.05,

    # Episode-final-stage dynamic reward curriculum.
    #
    # Stage graph:
    #   S0 -> S1: true 3-D grasp error < 0.15 m for 3 consecutive HL steps.
    #   S1 -> S0: true 3-D grasp error > 0.15 m for 3 consecutive HL steps.
    #   S1 -> S2: simulation-confirmed grasp.
    #   S2 is absorbing; losing grasp for 2 consecutive HL steps terminates.
    #
    # At reset, the FINAL stage reached at episode end determines which reward
    # group is reinforced for the next episode. Every curriculum reward has its
    # own lower/upper bound, while all rewards share one multiplicative factor.
    "dynamic_reward_scale_multiplier": 1.1,

    # Episode success / reward-scale freeze criterion.
    # An episode is latched successful once, while in curriculum Stage 2,
    # the 3-D object-to-retrieval-target error stays below 0.15 m for
    # 3 consecutive HL steps. Five consecutive valid successful episodes
    # permanently freeze that environment's dynamic reward scales.
    "retrieval_success_error_threshold": 0.15,
    "retrieval_success_consecutive_steps": 3,
    "dynamic_reward_freeze_after_consecutive_success_episodes": 5,

    "curriculum_s0_to_s1_grasp_error_threshold": 0.20,
    "curriculum_s0_to_s1_consecutive_steps": 3,
    "curriculum_s1_to_s0_grasp_error_threshold": 0.25,
    "curriculum_s1_to_s0_consecutive_steps": 3,
    "curriculum_s2_grasp_lost_consecutive_steps": 2,

    # S0 reward group.
    "grasp_align_coarse_weight_initial": 2.0,
    "grasp_align_coarse_weight_min": 0.5,
    "grasp_align_coarse_weight_max": 2.0,

    # S1 reward group.
    "grasp_align_fine_weight_initial": 2.0,
    "grasp_align_fine_weight_min": 2.0,
    "grasp_align_fine_weight_max": 10.0,
    "gripper_action_target_weight_initial": 0.05,
    "gripper_action_target_weight_min": 0.05,
    "gripper_action_target_weight_max": 1.0,
    "grasp_success_bonus_weight_initial": 50.0,
    "grasp_success_bonus_weight_min": 50.0,
    "grasp_success_bonus_weight_max": 250.0,

    # S0/S1 shared active-perception reward.
    # Front optical-camera center copied exactly from b2wz1.urdf:
    #   f_oc_link is a fixed child of base_link
    #   origin xyz="0.3993 0 -0.01576"
    #
    # Reward definition:
    #   minimize the full 3-D angle between
    #       (1) robot heading = base_link local +X axis expressed in world frame
    #       (2) front-camera-center -> object-center line of sight.
    #
    # Runtime gating:
    #   active only while deployable grasp_confidence_proxy == 0
    #   inactive immediately once grasp_confidence_proxy == 1
    #
    # In the nominal stage progression this corresponds to active perception
    # during the pre-grasp S0/S1 behavior and disabling it once the deployable
    # grasp proxy reports a grasp. Note that proxy==1 can occur while curriculum
    # stage is still S1 if simulation-only dual contact has not yet confirmed S2.
    #
    # Special dynamic-curriculum rule:
    #   final episode stage S0/S1 -> keep active-perception weight unchanged
    #   final episode stage S2    -> divide active-perception weight by the shared
    #                                curriculum multiplier, down to its minimum.
    #
    # Start at the maximum so S2 completion can meaningfully anneal this auxiliary
    # shaping reward from 1.5 toward 0.5.
    "front_optical_camera_pos_base": [0.3993, 0.0, -0.01576],
    "active_perception_exp_std": math.radians(30.0),
    "active_perception_weight_initial": 0.0,
    "active_perception_weight_min": 0.0,
    "active_perception_weight_max": 0.0,

    # S2 retrieval reward group.
    # At reset, sample one persistent retrieval target per episode:
    #   XY: uniformly by area inside a 0.5 m-radius disk centered at the
    #       robot's initialized world-frame x/y position.
    #   Z : independently and uniformly in world-frame [0.30, 0.80] m.
    #
    # The target remains fixed in world frame for the whole episode.
    "retrieval_target_radius": 0.50,
    "retrieval_target_z_range_w": [0.30, 0.80],
    "retrieval_target_exp_std": 0.35,
    "retrieval_target_weight_initial": 5.0,
    "retrieval_target_weight_min": 5.0,
    "retrieval_target_weight_max": 20.0,

    # Stage-2 base-heading reward:
    # minimize the planar angle between the robot base +X heading and the
    # current base->retrieval-target direction.
    "retrieval_heading_exp_std": math.radians(45.0),
    "retrieval_heading_weight_initial": 1.0,
    "retrieval_heading_weight_min": 1.0,
    "retrieval_heading_weight_max": 1.0,

    # Stage-2 raw gripper-close action reward.
    # Reward factor is the policy's continuous close confidence:
    #   close_confidence = 0.5 * (raw_gripper_action + 1)
    # so raw action -1 -> 0, 0 -> 0.5, +1 -> 1.
    # This trains the policy itself to keep commanding close even though the
    # optional Stage-2 execution heuristic may physically force the gripper closed.
    "stage2_gripper_close_action_weight_initial": 0.05,
    "stage2_gripper_close_action_weight_min": 0.50,
    "stage2_gripper_close_action_weight_max": 0.50,

    # Binary penalty when either gripper link contacts the ground.
    # GPU contact filtering against the static terrain collider is unsupported,
    # so use the existing unfiltered robot contact sensor and require both:
    #   contact force > threshold
    #   link origin height <= ground_z + height_threshold
    "gripper_ground_contact_force_threshold": 1.0,
    "gripper_ground_contact_height_threshold": 0.10,
    "gripper_ground_contact_penalty_weight": 0.0,

    # High-level normalized base-action smoothness penalties.
    "base_action_rate_penalty_weight": 0.01,
    # Second-order smoothness on normalized base actions:
    #   penalty = -weight * ||a_t - 2 a_{t-1} + a_{t-2}||^2
    "base_action_second_order_penalty_weight": 0.005,

    # Small fixed penalties on the measured gripperStator rigid-body velocity.
    # These use world-frame velocity norms, which are frame-invariant.
    # Penalty form:
    #   -lin_weight * ||v_ee||^2 - ang_weight * ||omega_ee||^2
    "end_effector_lin_vel_penalty_weight": 0.01,
    "end_effector_ang_vel_penalty_weight": 0.01,

    # Penalize the magnitude of the actually executed base velocity command:
    #   penalty = -weight * sum(abs([vx_cmd, vy_cmd, wz_cmd]))
    # This becomes zero when the optional Stage-2 base-still override is active.
    "base_velocity_command_abs_penalty_weight": 0.0,

    # Penalize cube planar motion while the gripper is commanded open.
    # A 0.02 m/s deadband removes tiny solver/numerical drift:
    #   effective_speed = relu(planar_speed - deadband)
    #   penalty = -weight * effective_speed^2
    "gripper_open_cube_planar_velocity_penalty_weight": 0.0,
    "gripper_open_cube_planar_velocity_deadband": 0.02,

    # Per-step gripper-action timing reward threshold.
    # Reward semantics:
    #   far  + open  -> 0
    #   far  + close -> penalty
    #   near + open  -> 0
    #   near + close -> reward
    # Only the positive side of the raw gripper action contributes to this
    # timing reward; open actions do not accumulate positive reward.
    "gripper_action_target_error_threshold": 0.08,

    # Deployable grasp-confidence proxy exposed to the actor:
    #   close command
    #   AND grasp error < 0.10 m
    #   AND gripper not fully closed
    #   AND actual gripper angle changes by at most 3 degrees per HL step.
    # Three consecutive true/false steps provide enter/exit hysteresis.
    "grasp_proxy_error_threshold": 0.10,
    # Closed target is 0 rad. Treat the gripper as not fully closed only when
    # the measured angle remains strictly below -5 degrees.
    "gripper_not_fully_closed_angle_threshold": math.radians(-5.0),
    "gripper_angle_hold_threshold": math.radians(3.0),
    "grasp_proxy_enter_steps": 3,
    "grasp_proxy_exit_steps": 3,


    # Contact threshold used only to build privileged critic observations.
    "gripper_cube_contact_force_threshold": 0.30,

    # Gripper command
    # Official gripper maximum opening angle is 90 degrees.
    # jointGripper uses 0 rad as closed and -pi/2 rad as fully open.
    # The robot asset/URDF joint limit must also allow [-pi/2, 0].
    # The high-level policy still outputs one scalar action for the gripper,
    # but the environment binarizes it before execution:
    #   action[-1] < 0  -> open
    #   action[-1] >= 0 -> close
    # Thus RL learns only the open/close timing, not a continuous gripper aperture.
    "gripper_open_pos": -math.pi / 2.0,
    "gripper_close_pos": 0.0,
    "gripper_binary_threshold": 0.0,

    # Reset domain randomization.
    # Base x/y/yaw are kept at their nominal reset values. Randomization is
    # restricted to vertical position, roll/pitch, small root velocities, and
    # modest joint-position scaling.
    "reset_base_pose_range": {
        "z": (0.0, 0.1),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
    },
    "reset_base_velocity_range": {
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.05, 0.05),
        "roll": (-0.05, 0.05),
        "pitch": (-0.05, 0.05),
        "yaw": (-0.05, 0.05),
    },
    # Keep the leg pose exactly at the robot configuration default.
    "reset_leg_joint_position_scale_range": (1.0, 1.0),
    # Apply only a small arm-pose perturbation around the robot configuration default.
    "reset_arm_joint_position_scale_range": (0.95, 1.05),

    # Episode
    "episode_length_s": 15.0,

    # Terminate when the true 3-D gripper-center/object distance remains
    # above 3.0 m for 3 consecutive high-level steps.
    "grasp_error_termination_threshold": 3.0,
    "grasp_error_termination_consecutive_steps": 3,

    # Base-link ground contact remains a safety termination only.
    "base_contact_body_patterns": [r"base_link"],
    "contact_force_threshold": 1.0,


    # Debug visualization for the legacy object / EE markers.
    "debug_vis": False,

    # Independent retrieval-target marker switch.
    # This is intentionally enabled during training and does not depend on debug_vis.
    "retrieval_target_marker_vis": True,
}

@configclass
class EventCfg:
    """Domain randomization terms applied at simulation startup.

    Root/joint reset randomization is implemented in ``_reset_idx`` because this
    task places the ball relative to the randomized initial robot yaw.
    """

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.40, 1.20),
            "dynamic_friction_range": (0.40, 1.20),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    ball_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "static_friction_range": (1.00, 2.00),
            "dynamic_friction_range": (1.00, 2.00),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-2.0, 2.0),
            "operation": "add",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "z": (-0.02, 0.02),
            },
        },
    )

    object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "mass_distribution_params": (0.10, 0.30),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

@configclass
class B2WZ1HLBallPickUpEnvCfg(DirectRLEnvCfg):
    """Stage-independent high-level Beta-PPO mobile-manipulation grasp environment."""

    env_params = ENV_PARAMS
    play_mode = False

    episode_length_s = ENV_PARAMS["episode_length_s"]

    # One HL step internally executes ll_steps_per_hl_step * ll_decimation sim steps.
    decimation = ENV_PARAMS["ll_steps_per_hl_step"] * ENV_PARAMS["ll_decimation"]

    # 9-D HL action:
    # [base_vx, base_vy, base_wz, delta_kp0_x, delta_kp0_y,
    #  delta_kp0_z, delta_ee_yaw, delta_ee_pitch, gripper_open_close]
    # Arm deltas are rebuilt from the current measured EE pose in PLB every HL step.
    action_space = 9

    # Actor frame (56-D):
    # root angular velocity in body frame (3)
    # projected_gravity_b(3)
    # leg/arm/gripper joint positions (12+6+1)
    # arm joint velocities (6)
    # noisy object-center position in the current full base/body frame (3)
    # noisy gripper local +X/+Y axes expressed in the current base/body frame (6)
    # noisy gripper-center position in the current base/body frame (3)
    # noisy retrieval-target position in the current full root/body frame (3)
    # previous HL action (9): effective normalized executed base action after
    # optional Stage-2 override, raw normalized arm action, and executed binary gripper action
    # deployable grasp-confidence proxy(1)
    obs_dim = 3 + 3 + 12 + 6 + 1 + 6 + 3 + 6 + 3 + 3 + 9 + 1

    # Critic keeps root linear velocity (3) in addition to the actor base frame,
    # then adds cube bottom/top centers in the current full base/body frame(3+3),
    # base height(1), two contact flags(1+1), object mass(1), and side length(1).
    critic_obs_dim = obs_dim + 3 + 3 + 3 + 1 + 1 + 1 + 1 + 1

    observation_space = obs_dim * ENV_PARAMS["obs_history"]
    state_space = critic_obs_dim * ENV_PARAMS["critic_obs_history"]

    viewer: ViewerCfg = ViewerCfg(
        eye=(6.0, 4.0, 4.0),
        lookat=(0.0, 0.0, 0.5),
        resolution=(1280, 720),
    )

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,
        render_interval=ENV_PARAMS["ll_decimation"],
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            # Use normal ground friction; very high friction can destabilize contacts.
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # Read the complete default root/joint configuration directly from
    # UNITREE_B2WZ1_CFG. The environment must not shift the frozen LL policy's
    # joint-position observation zero or action-target baseline.
    robot: ArticulationCfg = UNITREE_B2WZ1_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/TennisBall",
        spawn=sim_utils.CuboidCfg(
            size=(ENV_PARAMS["cube_side"], ENV_PARAMS["cube_side"], ENV_PARAMS["cube_side"]),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=ENV_PARAMS["ball_mass"],
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 1.0, 0.1),
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.0, 0.0, ENV_PARAMS["ground_z"] + ENV_PARAMS["object_half_extent"]),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # Filtered sensors provide cube-specific contact flags to the critic.
    gripper_stator_cube_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/gripperStator",
        history_length=3,
        update_period=0.005,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/TennisBall"],
    )

    gripper_mover_cube_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/gripperMover",
        history_length=3,
        update_period=0.005,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/TennisBall"],
    )

    object_markers_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/ObjectMarkers",
        markers={
            "object": sim_utils.SphereCfg(
                radius=ENV_PARAMS["object_half_extent"],
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),
            )
        },
    )

    ee_target_markers_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/EETargetMarkers",
        markers={
            "ee_target": sim_utils.SphereCfg(
                radius=0.04,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            )
        },
    )

    retrieval_target_markers_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/RetrievalTargetMarkers",
        markers={
            "retrieval_target": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)),
            )
        },
    )

    add_noise = ENV_PARAMS["obs_noise"]
    events: EventCfg = EventCfg()
    debug_vis = ENV_PARAMS["debug_vis"]

class B2WZ1HLBallPickUpEnv(DirectRLEnv):
    cfg: B2WZ1HLBallPickUpEnvCfg

    def _joint_ids_in_order(self, joint_names: list[str]) -> list[int]:
        """Resolve joint ids while preserving the given joint_names order."""
        name_to_id = {name: i for i, name in enumerate(self._robot.joint_names)}

        missing = [name for name in joint_names if name not in name_to_id]
        if missing:
            raise RuntimeError(
                f"Missing joints {missing}. Available joints are: {self._robot.joint_names}"
            )

        return [name_to_id[name] for name in joint_names]

    def _body_id_scalar(self, body_id) -> int:
        """Convert IsaacLab body id output to a Python int."""
        if isinstance(body_id, torch.Tensor):
            return int(body_id.flatten()[0].item())
        if isinstance(body_id, (list, tuple)):
            body_id = body_id[0]
            if isinstance(body_id, torch.Tensor):
                return int(body_id.item())
            return int(body_id)
        return int(body_id)

    def __init__(self, cfg: B2WZ1HLBallPickUpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        p = self.cfg.env_params

        self._base_cmd_scale = torch.tensor(
            p["base_cmd_scale"],
            dtype=torch.float32,
            device=self.device,
        )

        self._neutral_kp0 = torch.tensor(
            p["neutral_kp0"],
            dtype=torch.float32,
            device=self.device,
        )

        # These orders must match the low-level exported policy obs/action order.
        self.leg_joint_names = [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        ]

        self.arm_joint_names = [
            "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
        ]

        # Low-level policy uses wheel semantics, but articulation names use foot_joint.
        self.wheel_joint_names = [
            "FL_foot_joint", "FR_foot_joint", "RL_foot_joint", "RR_foot_joint",
        ]

        self.gripper_joint_names = ["jointGripper"]

        # Low-level policy action order: leg + arm + wheel.
        # Gripper is controlled directly by high-level policy, not by frozen LL policy.
        self.policy_action_names = (
            self.leg_joint_names
            + self.arm_joint_names
            + self.wheel_joint_names
        )

        # End-effector reference body used by the frozen low-level policy.
        # Keep this as gripperStator.
        self.ee_body_name = "gripperStator"

        # Fixed 5 cm grasp-pocket reference measured from the real gripperStator.
        # Unlike a mover-attached marker, this center does not drift during closure.
        self.gripper_center_body_name = "gripperStator"
        self.gripper_center_body_id, _ = self._robot.find_bodies(self.gripper_center_body_name)
        self.gripper_center_body_id_scalar = self._body_id_scalar(self.gripper_center_body_id)

        # Keep actual base_link body pose only for diagnostics.  Task reset,
        # reward and base gating use PLB to match the frozen low-level command
        # frame.
        self.base_link_body_name = "base_link"
        self.base_link_body_id, _ = self._robot.find_bodies(self.base_link_body_name)
        self.base_link_body_id_scalar = self._body_id_scalar(self.base_link_body_id)

        # Fixed front-camera center from b2wz1.urdf (base_link -> f_oc_link).
        # The active-perception reward uses this translation to construct the true
        # camera-center -> object-center LOS. Camera orientation is not needed for
        # the current robot-heading-vs-LOS objective.
        self._front_optical_camera_pos_base = torch.tensor(
            p["front_optical_camera_pos_base"],
            dtype=torch.float32,
            device=self.device,
        )

        self._gripper_center_offset_local = torch.tensor(
            p["gripper_center_offset_local"],
            dtype=torch.float32,
            device=self.device,
        )

        self.leg_ids = self._joint_ids_in_order(self.leg_joint_names)
        self.arm_ids = self._joint_ids_in_order(self.arm_joint_names)
        self.wheel_ids = self._joint_ids_in_order(self.wheel_joint_names)
        self.gripper_ids = self._joint_ids_in_order(self.gripper_joint_names)
        self.policy_action_ids = self._joint_ids_in_order(self.policy_action_names)

        self.ee_body_id, _ = self._robot.find_bodies(self.ee_body_name)
        self.ee_body_id_scalar = self._body_id_scalar(self.ee_body_id)

        # Articulation body ids are used for the gripper-link height guard.
        self._gripper_stator_body_id, _ = self._robot.find_bodies("gripperStator")
        self._gripper_mover_body_id, _ = self._robot.find_bodies("gripperMover")
        self._gripper_stator_body_id_scalar = self._body_id_scalar(
            self._gripper_stator_body_id
        )
        self._gripper_mover_body_id_scalar = self._body_id_scalar(
            self._gripper_mover_body_id
        )

        # Base-link contact is retained only as a safety termination.
        contact_body_names = self._contact_sensor.body_names
        base_contact_body_ids = [
            body_id
            for body_id, body_name in enumerate(contact_body_names)
            if any(re.fullmatch(pattern, body_name) for pattern in p["base_contact_body_patterns"])
        ]
        if len(base_contact_body_ids) == 0:
            raise RuntimeError(
                f"No base contact bodies matched patterns {p['base_contact_body_patterns']}. "
                f"Available contact sensor bodies are: {contact_body_names}"
            )
        self._base_contact_body_ids = torch.tensor(
            base_contact_body_ids, dtype=torch.long, device=self.device
        )

        contact_name_to_id = {
            body_name: body_id
            for body_id, body_name in enumerate(contact_body_names)
        }
        missing_gripper_contact_bodies = [
            name
            for name in ("gripperStator", "gripperMover")
            if name not in contact_name_to_id
        ]
        if missing_gripper_contact_bodies:
            raise RuntimeError(
                "Generic robot contact sensor is missing gripper bodies "
                f"{missing_gripper_contact_bodies}. Available bodies: "
                f"{contact_body_names}"
            )
        self._gripper_stator_contact_body_id = contact_name_to_id["gripperStator"]
        self._gripper_mover_contact_body_id = contact_name_to_id["gripperMover"]

        self._base_contact = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._gripper_stator_cube_contact = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._gripper_mover_cube_contact = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._gripper_stator_ground_contact = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._gripper_mover_ground_contact = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._gripper_ground_contact = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        self.num_leg = len(self.leg_ids)
        self.num_arm = len(self.arm_ids)
        self.num_wheel = len(self.wheel_ids)
        self.num_gripper = len(self.gripper_ids)

        assert self.num_leg == 12, f"Expected 12 leg joints, got {self.num_leg}"
        assert self.num_arm == 6, f"Expected 6 arm joints, got {self.num_arm}"
        assert self.num_wheel == 4, f"Expected 4 wheel joints, got {self.num_wheel}"
        assert self.num_gripper == 1, f"Expected 1 gripper joint, got {self.num_gripper}"
        assert len(self.policy_action_ids) == 22, (
            f"Expected 22 policy action joints, got {len(self.policy_action_ids)}"
        )

        self.ll_action_dim = self.num_leg + self.num_arm + self.num_wheel

        # LL obs frame:
        # root_ang_vel_b(3)
        # projected_gravity_b(3)
        # base_velocity_cmd(3)
        # ee_kp_cmd_plb(9)
        # joint_pos_leg_rel(12)
        # joint_pos_arm_rel(6)
        # joint_vel_leg(12)
        # joint_vel_arm(6)
        # joint_vel_wheel(4)
        # last_ll_action(22)
        self.ll_frame_dim = (
            3 + 3 + 3 + 9
            + self.num_leg + self.num_arm
            + self.num_leg + self.num_arm + self.num_wheel
            + self.ll_action_dim
        )

        assert self.ll_action_dim == 22, f"Expected LL action dim 22, got {self.ll_action_dim}"
        assert self.ll_frame_dim == 80, f"Expected LL obs frame dim 80, got {self.ll_frame_dim}"

        self._ll_policy = LLPolicyWrapper(
            task=p["ll_task"],
            experiment_name=p["ll_experiment_name"],
            load_run=p["ll_load_run"],
            checkpoint=p["ll_load_checkpoint"],
            device=str(self.device),
        )

        self._ll_obs_history = torch.zeros(
            self.num_envs,
            p["ll_obs_history"],
            self.ll_frame_dim,
            device=self.device,
        )

        self._ll_actions = torch.zeros(self.num_envs, self.ll_action_dim, device=self.device)

        self._hl_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_hl_actions = torch.zeros_like(self._hl_actions)
        self._prev_prev_hl_actions = torch.zeros_like(self._hl_actions)

        self._obs_history = torch.zeros(
            self.num_envs,
            p["obs_history"],
            self.cfg.obs_dim,
            device=self.device,
        )

        self._critic_history = torch.zeros(
            self.num_envs,
            p["critic_obs_history"],
            self.cfg.critic_obs_dim,
            device=self.device,
        )

        self._base_velocity_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self._ee_kp_cmd_plb = torch.zeros(self.num_envs, 9, device=self.device)

        # PLB EE command produced from the current measured gripperStator pose
        # plus the current high-level delta action. It is recomputed every HL
        # step and is never integrated from the previous command.
        self._kp0_cmd = self._neutral_kp0.unsqueeze(0).repeat(self.num_envs, 1)
        self._prev_kp0_cmd = self._kp0_cmd.clone()
        self._ee_yaw_cmd = torch.full(
            (self.num_envs,), p["neutral_ee_yaw"], dtype=torch.float32, device=self.device
        )
        self._ee_pitch_cmd = torch.full(
            (self.num_envs,), p["neutral_ee_pitch"], dtype=torch.float32, device=self.device
        )
        self._ee_roll_cmd = torch.full(
            (self.num_envs,), p["fixed_ee_roll"], dtype=torch.float32, device=self.device
        )
        self._prev_ee_yaw_cmd = self._ee_yaw_cmd.clone()
        self._prev_ee_pitch_cmd = self._ee_pitch_cmd.clone()

        self._gripper_cmd_pos = torch.full(
            (self.num_envs, 1),
            p["gripper_open_pos"],
            dtype=torch.float32,
            device=self.device,
        )
        self._gripper_cmd_norm = torch.full(
            (self.num_envs, 1),
            -1.0,
            dtype=torch.float32,
            device=self.device,
        )
        # Binary command executed during the preceding HL step.
        # -1 means open, +1 means close.
        self._prev_gripper_cmd_norm = self._gripper_cmd_norm.clone()

        self._raw_gripper_action = torch.full(
            (self.num_envs,),
            -1.0,
            dtype=torch.float32,
            device=self.device,
        )
        # Continuous confidence implied by the raw gripper action:
        # 0 means strongly open, 1 means strongly close. This is not a
        # physical or contact-based grasp-confidence estimate.
        self._gripper_close_confidence = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )

        # Actual gripper joint position at the preceding high-level step.
        # Used by the privileged grasp-state estimator.
        self._prev_gripper_joint_pos = torch.full(
            (self.num_envs,),
            float(p["gripper_open_pos"]),
            dtype=torch.float32,
            device=self.device,
        )
        self._gripper_angle_delta = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self._gripper_not_fully_closed = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._gripper_angle_holding = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

        self._object_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._object_pos_plb = torch.zeros(self.num_envs, 3, device=self.device)
        self._object_center_pos_base = torch.zeros(self.num_envs, 3, device=self.device)
        self._object_height = torch.zeros(self.num_envs, device=self.device)

        # Per-episode retrieval target. The target is persistent in world frame and
        # transformed into the current robot root/body frame for observations.
        self._retrieval_target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._retrieval_target_pos_base = torch.zeros(self.num_envs, 3, device=self.device)

        self._object_grippercenter_error_plb = torch.zeros(self.num_envs, 3, device=self.device)
        self._object_pos_grippercenter = torch.zeros(self.num_envs, 3, device=self.device)

        # Object face centers are retained in PLB for task logic and also
        # expressed in the full base/body frame for privileged critic input.
        self._object_bottom_center_pos_plb = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._object_top_center_pos_plb = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._object_bottom_center_pos_base = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._object_top_center_pos_base = torch.zeros(
            self.num_envs, 3, device=self.device
        )

        self._gripper_center_pos_plb = torch.zeros(self.num_envs, 3, device=self.device)
        self._gripper_center_pos_base = torch.zeros(self.num_envs, 3, device=self.device)
        self._gripper_orientation_base = torch.zeros(self.num_envs, 6, device=self.device)

        # Task-target and geometry buffers.
        self._gripper_center_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._gripper_stator_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._grasp_error = torch.zeros(self.num_envs, device=self.device)

        self._grasp_error_oob_steps = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self._grasp_error_oob_terminated = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

        # Deployable grasp-confidence proxy exposed to the actor.
        self._grasp_confidence_proxy = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._prev_grasp_confidence_proxy = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._grasp_proxy_candidate = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._grasp_proxy_enter_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._grasp_proxy_exit_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # Simulation-only confirmation additionally requires dual cube contact.
        # This state gates target/base rewards but is not exposed to the actor.
        self._stage2 = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._prev_stage2 = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._grasp_true_candidate = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Episode stage machine and per-environment reward curriculum.
        # Actor and critic observations are intentionally unchanged.
        self._curriculum_stage = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._curriculum_s0_to_s1_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._curriculum_s1_to_s0_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._curriculum_s2_grasp_lost_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._curriculum_s2_grasp_lost_terminated = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._curriculum_s0_to_s1_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._curriculum_s1_to_s0_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._curriculum_s1_to_s2_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._grasp_bonus_awarded = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Consecutive complete episodes are counted independently per environment.
        # Once the threshold is reached, that environment's reward scales are frozen.
        # Per-episode retrieval success:
        # Stage 2 AND retrieval error < threshold for N consecutive HL steps.
        # _episode_retrieval_success is latched True for the rest of the episode.
        self._retrieval_success_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._episode_retrieval_success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Consecutive successful valid episodes are counted independently per env.
        # Once the configured threshold is reached, reward scales are frozen forever.
        self._curriculum_complete_episode_streak = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._curriculum_reward_scales_frozen = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        self._curriculum_reward_names = (
            "grasp_align_coarse",
            "grasp_align_fine",
            "gripper_action_target",
            "grasp_success_bonus",
            "retrieval_target",
            "retrieval_heading",
            "stage2_gripper_close_action",
            "active_perception",
        )
        # Stage ownership for the standard curriculum rewards.
        # active_perception uses -1 because it is a special S0/S1-shared reward:
        # it is not reinforced by either S0 or S1 final stage and is annealed only
        # when an episode reaches final Stage 2.
        self._curriculum_reward_stage = torch.tensor(
            [0, 1, 1, 1, 2, 2, 2, -1],
            dtype=torch.long,
            device=self.device,
        )
        self._active_perception_reward_idx = self._curriculum_reward_names.index(
            "active_perception"
        )
        self._curriculum_reward_weights = torch.tensor(
            [
                p["grasp_align_coarse_weight_initial"],
                p["grasp_align_fine_weight_initial"],
                p["gripper_action_target_weight_initial"],
                p["grasp_success_bonus_weight_initial"],
                p["retrieval_target_weight_initial"],
                p["retrieval_heading_weight_initial"],
                p["stage2_gripper_close_action_weight_initial"],
                p["active_perception_weight_initial"],
            ],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        self._curriculum_reward_weight_min = torch.tensor(
            [
                p["grasp_align_coarse_weight_min"],
                p["grasp_align_fine_weight_min"],
                p["gripper_action_target_weight_min"],
                p["grasp_success_bonus_weight_min"],
                p["retrieval_target_weight_min"],
                p["retrieval_heading_weight_min"],
                p["stage2_gripper_close_action_weight_min"],
                p["active_perception_weight_min"],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self._curriculum_reward_weight_max = torch.tensor(
            [
                p["grasp_align_coarse_weight_max"],
                p["grasp_align_fine_weight_max"],
                p["gripper_action_target_weight_max"],
                p["grasp_success_bonus_weight_max"],
                p["retrieval_target_weight_max"],
                p["retrieval_heading_weight_max"],
                p["stage2_gripper_close_action_weight_max"],
                p["active_perception_weight_max"],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        self._prepare_rewards()
        self.set_debug_vis(self.cfg.debug_vis)

        # Retrieval-target visualization is independent of the legacy debug_vis
        # switch so it can stay visible during training.
        if bool(p["retrieval_target_marker_vis"]):
            self._retrieval_target_markers = VisualizationMarkers(
                self.cfg.retrieval_target_markers_cfg
            )
            self._retrieval_target_markers.set_visibility(True)

        if self.cfg.play_mode:
            print(
                "[B2WZ1HL] neutral kp0/yaw/pitch/roll:",
                self._neutral_kp0.tolist(),
                float(p["neutral_ee_yaw"]),
                float(p["neutral_ee_pitch"]),
                float(p["fixed_ee_roll"]),
            )
            print(
                "[B2WZ1HL] robot-cfg default arm joint positions:",
                self._robot.data.default_joint_pos[0, self.arm_ids],
            )
            print("[B2WZ1HL] jointGripper ids:", self.gripper_ids)
            print(
                "[B2WZ1HL] jointGripper soft limits:",
                self._robot.data.soft_joint_pos_limits[0, self.gripper_ids],
            )
            print("[B2WZ1HL] base contact terminal bodies:", [contact_body_names[i] for i in base_contact_body_ids])
    
    # Scene setup
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._ball = RigidObject(self.cfg.ball)
        self.scene.rigid_objects["ball"] = self._ball

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self._gripper_stator_cube_contact_sensor = ContactSensor(
            self.cfg.gripper_stator_cube_contact_sensor
        )
        self.scene.sensors["gripper_stator_cube_contact_sensor"] = (
            self._gripper_stator_cube_contact_sensor
        )

        self._gripper_mover_cube_contact_sensor = ContactSensor(
            self.cfg.gripper_mover_cube_contact_sensor
        )
        self.scene.sensors["gripper_mover_cube_contact_sensor"] = (
            self._gripper_mover_cube_contact_sensor
        )

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.75, 0.75, 0.75),
        )
        light_cfg.func("/World/Light", light_cfg)

    # Helpers
    def _add_observation_noise(self, tensor: torch.Tensor, noise_scale: float) -> torch.Tensor:
        if self.cfg.add_noise and not self.cfg.play_mode:
            return tensor + (torch.rand_like(tensor) * 2.0 * noise_scale - noise_scale)
        return tensor

    @staticmethod
    def _scale_action_to_range(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
        return low + 0.5 * (x + 1.0) * (high - low)

    @staticmethod
    def _quat_wxyz_to_euler_xyz(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert normalized wxyz quaternions to XYZ roll, pitch, yaw."""
        quat = quat / torch.clamp(torch.norm(quat, dim=-1, keepdim=True), min=1.0e-8)
        w, x, y, z = quat.unbind(dim=-1)

        sin_roll_cos_pitch = 2.0 * (w * x + y * z)
        cos_roll_cos_pitch = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)

        sin_pitch = 2.0 * (w * y - z * x)
        pitch = torch.asin(torch.clamp(sin_pitch, -1.0, 1.0))

        sin_yaw_cos_pitch = 2.0 * (w * z + x * y)
        cos_yaw_cos_pitch = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)
        return roll, pitch, yaw

    def _get_actual_ee_pose_plb(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return measured gripperStator position, yaw and pitch in PLB."""
        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w
        ee_pos_w = self._robot.data.body_pos_w[:, self.ee_body_id_scalar]
        ee_quat_w = self._robot.data.body_quat_w[:, self.ee_body_id_scalar]

        ee_pos_plb = transform_points_w_to_plb(
            root_pos_w=root_pos_w,
            root_quat_w=root_quat_w,
            points_w=ee_pos_w.unsqueeze(1),
            ground_z=float(self.cfg.env_params["ground_z"]),
        )[:, 0]

        # PLB uses the gravity-leveled/yaw heading of the robot. Express the EE
        # orientation relative to that heading, then extract XYZ pitch/yaw.
        plb_yaw_quat_w = math_utils.yaw_quat(root_quat_w)
        ee_quat_plb = math_utils.quat_mul(
            math_utils.quat_inv(plb_yaw_quat_w),
            ee_quat_w,
        )
        _, ee_pitch_plb, ee_yaw_plb = self._quat_wxyz_to_euler_xyz(ee_quat_plb)
        return ee_pos_plb, ee_yaw_plb, ee_pitch_plb

    @staticmethod
    def _exp_square_reward(error: torch.Tensor, std: float) -> torch.Tensor:
        return torch.exp(-torch.square(error / max(std, 1.0e-6)))

    @staticmethod
    def _exp_abs_reward(error: torch.Tensor, std: float) -> torch.Tensor:
        """Return exp(-abs(error) / std)."""
        return torch.exp(-torch.abs(error) / max(std, 1.0e-6))

    def _get_object_sampling_params(self) -> tuple[float, float, float]:
        """Return fixed polar object sampling parameters.

        The sampling range is intentionally kept unchanged:
            r = U(object_sampling_r_range)
            theta = U(-object_sampling_theta_max, object_sampling_theta_max)
        """
        p = self.cfg.env_params
        base_r = p["object_sampling_r_range"]
        return float(base_r[0]), float(base_r[1]), float(p["object_sampling_theta_max"])

    def _sample_object_xy_plb(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample object x/y in the PLB frame from a pregrasp-centered polar sector."""
        p = self.cfg.env_params
        r_min, r_max, theta_max = self._get_object_sampling_params()

        r = torch.empty(num_samples, device=self.device).uniform_(r_min, r_max)
        theta = torch.empty(num_samples, device=self.device).uniform_(-theta_max, theta_max)

        object_x = float(p["object_sampling_center_x"]) + r * torch.cos(theta)
        object_y = float(p.get("object_sampling_center_y", 0.0)) + r * torch.sin(theta)
        return object_x, object_y

    def _compute_object_pos_leveled_base_and_axes_w(
        self,
        object_pos_w: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Express the object in a gravity-leveled local base frame.

        The leveled frame shares the robot root/base_link origin. Its +Z axis is
        opposite projected gravity; its +X axis is the base forward axis projected
        onto the local horizontal plane. It therefore requires IMU gravity, FK and
        camera/base extrinsics on hardware, but not a world-frame base pose.
        """
        eps = 1.0e-6
        if env_ids is None:
            root_pos_w = self._robot.data.root_pos_w
            root_quat_w = self._robot.data.root_quat_w
            gravity_down_b = self._robot.data.projected_gravity_b
        else:
            root_pos_w = self._robot.data.root_pos_w[env_ids]
            root_quat_w = self._robot.data.root_quat_w[env_ids]
            gravity_down_b = self._robot.data.projected_gravity_b[env_ids]

        object_rel_w = object_pos_w - root_pos_w
        object_pos_base = math_utils.quat_apply(
            math_utils.quat_inv(root_quat_w), object_rel_w
        )

        gravity_down_b = gravity_down_b / torch.clamp(
            torch.norm(gravity_down_b, dim=-1, keepdim=True), min=eps
        )
        z_leveled_b = -gravity_down_b

        num_samples = object_pos_w.shape[0]
        base_forward_b = torch.tensor(
            [1.0, 0.0, 0.0], dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(num_samples, 1)
        x_leveled_b = base_forward_b - torch.sum(
            base_forward_b * z_leveled_b, dim=-1, keepdim=True
        ) * z_leveled_b
        x_leveled_b = x_leveled_b / torch.clamp(
            torch.norm(x_leveled_b, dim=-1, keepdim=True), min=eps
        )
        y_leveled_b = torch.cross(z_leveled_b, x_leveled_b, dim=-1)
        y_leveled_b = y_leveled_b / torch.clamp(
            torch.norm(y_leveled_b, dim=-1, keepdim=True), min=eps
        )
        x_leveled_b = torch.cross(y_leveled_b, z_leveled_b, dim=-1)

        rot_base_from_leveled = torch.stack(
            [x_leveled_b, y_leveled_b, z_leveled_b], dim=-1
        )
        object_pos_leveled = torch.bmm(
            rot_base_from_leveled.transpose(1, 2),
            object_pos_base.unsqueeze(-1),
        ).squeeze(-1)

        x_leveled_w = math_utils.quat_apply(root_quat_w, x_leveled_b)
        y_leveled_w = math_utils.quat_apply(root_quat_w, y_leveled_b)
        z_leveled_w = math_utils.quat_apply(root_quat_w, z_leveled_b)
        return object_pos_leveled, x_leveled_w, y_leveled_w, z_leveled_w

    def _neutral_ee_keypoints_plb(self, num: int | None = None) -> torch.Tensor:
        """Build neutral EE keypoints in PLB frame."""
        p = self.cfg.env_params

        if num is None:
            num = self.num_envs

        neutral_kp0 = self._neutral_kp0.unsqueeze(0).repeat(num, 1)
        neutral_yaw = torch.ones(num, device=self.device) * p["neutral_ee_yaw"]
        neutral_pitch = torch.ones(num, device=self.device) * p["neutral_ee_pitch"]
        neutral_roll = torch.full(
            (num,), float(p["fixed_ee_roll"]), dtype=torch.float32, device=self.device
        )

        return build_keypoints_from_kp0_yaw_pitch_plb(
            kp0=neutral_kp0,
            yaw=neutral_yaw,
            pitch=neutral_pitch,
            roll=neutral_roll,
            kp_dx=p["kp_dx"],
            kp_dz=p["kp_dz"],
        )

    # Debug visualization
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_object_markers"):
                self._object_markers = VisualizationMarkers(self.cfg.object_markers_cfg)
                self._ee_target_markers = VisualizationMarkers(self.cfg.ee_target_markers_cfg)

            self._object_markers.set_visibility(True)
            self._ee_target_markers.set_visibility(True)
        else:
            if hasattr(self, "_object_markers"):
                self._object_markers.set_visibility(False)
                self._ee_target_markers.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self._robot.is_initialized:
            return
        if not hasattr(self, "_object_markers"):
            return

        object_pos = self._object_pos_w.clone()
        gripper_center_pos = self._gripper_center_pos_w.clone()

        self._object_markers.visualize(translations=object_pos)
        self._ee_target_markers.visualize(translations=gripper_center_pos)

    # High-level action decoding
    def _pre_physics_step(self, actions: torch.Tensor):
        """Decode the full 9-D Beta action into base, EE and gripper commands."""
        self._prev_prev_hl_actions[:] = self._prev_hl_actions
        self._prev_hl_actions[:] = self._hl_actions
        self._prev_gripper_cmd_norm[:] = self._gripper_cmd_norm
        self._prev_kp0_cmd[:] = self._kp0_cmd
        self._prev_ee_yaw_cmd[:] = self._ee_yaw_cmd
        self._prev_ee_pitch_cmd[:] = self._ee_pitch_cmd

        self._hl_actions[:] = actions.clamp(-1.0, 1.0)
        self._decode_hl_action_to_ll_commands(self._hl_actions)

    def _decode_hl_action_to_ll_commands(self, actions: torch.Tensor):
        """Decode measured-EE-anchored PLB arm deltas and binarize the gripper."""
        p = self.cfg.env_params

        self._base_velocity_cmd[:] = actions[:, 0:3] * self._base_cmd_scale


        actual_ee_pos_plb, actual_ee_yaw_plb, actual_ee_pitch_plb = (
            self._get_actual_ee_pose_plb()
        )
        kp0_delta_scale = torch.tensor(
            p["kp0_delta_scale"], dtype=torch.float32, device=self.device
        ).reshape(1, 3)

        self._kp0_cmd[:] = actual_ee_pos_plb + actions[:, 3:6] * kp0_delta_scale
        self._kp0_cmd[:, 0].clamp_(float(p["kp0_x_range"][0]), float(p["kp0_x_range"][1]))
        self._kp0_cmd[:, 1].clamp_(float(p["kp0_y_range"][0]), float(p["kp0_y_range"][1]))
        self._kp0_cmd[:, 2].clamp_(float(p["kp0_z_range"][0]), float(p["kp0_z_range"][1]))

        self._ee_yaw_cmd[:] = actual_ee_yaw_plb + actions[:, 6] * float(
            p["ee_yaw_delta_scale"]
        )
        self._ee_pitch_cmd[:] = actual_ee_pitch_plb + actions[:, 7] * float(
            p["ee_pitch_delta_scale"]
        )
        self._ee_yaw_cmd.clamp_(float(p["ee_yaw_range"][0]), float(p["ee_yaw_range"][1]))
        self._ee_pitch_cmd.clamp_(float(p["ee_pitch_range"][0]), float(p["ee_pitch_range"][1]))
        self._ee_roll_cmd.fill_(float(p["fixed_ee_roll"]))

        self._ee_kp_cmd_plb[:] = build_keypoints_from_kp0_yaw_pitch_plb(
            kp0=self._kp0_cmd,
            yaw=self._ee_yaw_cmd,
            pitch=self._ee_pitch_cmd,
            roll=self._ee_roll_cmd,
            kp_dx=p["kp_dx"],
            kp_dz=p["kp_dz"],
        )

        # Binary gripper command is executed directly every HL step.
        # Positive action closes; negative action opens.
        self._raw_gripper_action[:] = actions[:, 8]
        close_commanded = self._raw_gripper_action > float(p["gripper_binary_threshold"])
        self._gripper_cmd_norm[:, 0] = torch.where(
            close_commanded,
            torch.ones_like(self._raw_gripper_action),
            -torch.ones_like(self._raw_gripper_action),
        )
        self._gripper_cmd_pos[:, 0] = torch.where(
            close_commanded,
            torch.full_like(self._raw_gripper_action, float(p["gripper_close_pos"])),
            torch.full_like(self._raw_gripper_action, float(p["gripper_open_pos"])),
        )
        self._gripper_close_confidence[:] = 0.5 * (self._raw_gripper_action + 1.0)

        # Optional execution overrides for environments already in curriculum S2.
        # Raw policy actions are retained for PPO and reward accounting.
        in_curriculum_stage2 = self._curriculum_stage == 2
        if bool(p["stage2_force_base_still_enabled"]):
            self._base_velocity_cmd[in_curriculum_stage2] = 0.0
        if bool(p["stage2_force_gripper_close_enabled"]):
            self._gripper_cmd_norm[in_curriculum_stage2, 0] = 1.0
            self._gripper_cmd_pos[in_curriculum_stage2, 0] = float(
                p["gripper_close_pos"]
            )

    def _update_grasp_stage(self) -> None:
        """Update deployable grasp proxy and simulation-confirmed grasp state."""
        p = self.cfg.env_params

        close_commanded = self._gripper_cmd_norm[:, 0] > 0.0
        dual_contact = (
            self._gripper_stator_cube_contact
            & self._gripper_mover_cube_contact
        )

        gripper_joint_pos = self._robot.data.joint_pos[:, self.gripper_ids[0]]
        self._gripper_angle_delta[:] = torch.abs(
            gripper_joint_pos - self._prev_gripper_joint_pos
        )

        not_fully_closed_threshold = float(
            p["gripper_not_fully_closed_angle_threshold"]
        )
        self._gripper_not_fully_closed[:] = (
            gripper_joint_pos < not_fully_closed_threshold
        )
        self._gripper_angle_holding[:] = (
            self._gripper_angle_delta
            < float(p["gripper_angle_hold_threshold"])
        )

        # Deployable proxy: command, camera/FK geometry and gripper encoder only.
        proxy_candidate = (
            close_commanded
            & (self._grasp_error < float(p["grasp_proxy_error_threshold"]))
            & self._gripper_not_fully_closed
            & self._gripper_angle_holding
        )
        self._grasp_proxy_candidate[:] = proxy_candidate

        not_confirmed = ~self._grasp_confidence_proxy
        self._grasp_proxy_enter_count[:] = torch.where(
            not_confirmed & proxy_candidate,
            self._grasp_proxy_enter_count + 1,
            torch.zeros_like(self._grasp_proxy_enter_count),
        )
        enter_proxy = not_confirmed & (
            self._grasp_proxy_enter_count >= int(p["grasp_proxy_enter_steps"])
        )
        self._grasp_confidence_proxy[enter_proxy] = True
        self._grasp_proxy_enter_count[enter_proxy] = 0

        confirmed = self._grasp_confidence_proxy
        self._grasp_proxy_exit_count[:] = torch.where(
            confirmed & (~proxy_candidate),
            self._grasp_proxy_exit_count + 1,
            torch.zeros_like(self._grasp_proxy_exit_count),
        )
        exit_proxy = confirmed & (
            self._grasp_proxy_exit_count >= int(p["grasp_proxy_exit_steps"])
        )
        self._grasp_confidence_proxy[exit_proxy] = False
        self._grasp_proxy_exit_count[exit_proxy] = 0

        # Simulation-only state used for reward conditioning.
        self._grasp_true_candidate[:] = proxy_candidate & dual_contact
        self._stage2[:] = self._grasp_confidence_proxy & dual_contact

        self._prev_gripper_joint_pos[:] = gripper_joint_pos

    def _update_curriculum_stage(self) -> None:
        """Update S0/S1/S2 with hysteresis and simulation-confirmed grasp."""
        p = self.cfg.env_params
        self._curriculum_s0_to_s1_event.zero_()
        self._curriculum_s1_to_s0_event.zero_()
        self._curriculum_s1_to_s2_event.zero_()

        stage0 = self._curriculum_stage == 0
        near = (
            self._grasp_error
            < float(p["curriculum_s0_to_s1_grasp_error_threshold"])
        )
        self._curriculum_s0_to_s1_count[:] = torch.where(
            stage0 & near,
            self._curriculum_s0_to_s1_count + 1,
            torch.zeros_like(self._curriculum_s0_to_s1_count),
        )
        enter_s1 = stage0 & (
            self._curriculum_s0_to_s1_count
            >= int(p["curriculum_s0_to_s1_consecutive_steps"])
        )
        self._curriculum_stage[enter_s1] = 1
        self._curriculum_s0_to_s1_event[enter_s1] = True
        self._curriculum_s0_to_s1_count[enter_s1] = 0

        stage1 = self._curriculum_stage == 1
        far = (
            self._grasp_error
            > float(p["curriculum_s1_to_s0_grasp_error_threshold"])
        )
        self._curriculum_s1_to_s0_count[:] = torch.where(
            stage1 & far,
            self._curriculum_s1_to_s0_count + 1,
            torch.zeros_like(self._curriculum_s1_to_s0_count),
        )
        return_s0 = stage1 & (
            self._curriculum_s1_to_s0_count
            >= int(p["curriculum_s1_to_s0_consecutive_steps"])
        )
        self._curriculum_stage[return_s0] = 0
        self._curriculum_s1_to_s0_event[return_s0] = True
        self._curriculum_s1_to_s0_count[return_s0] = 0

        # Simulation-confirmed grasp produces the only S1 -> S2 transition.
        stage1 = self._curriculum_stage == 1
        enter_s2 = stage1 & self._stage2
        self._curriculum_stage[enter_s2] = 2
        self._curriculum_s1_to_s2_event[enter_s2] = True

        if bool(p["stage2_force_base_still_enabled"]):
            self._base_velocity_cmd[enter_s2] = 0.0
        if bool(p["stage2_force_gripper_close_enabled"]):
            self._gripper_cmd_norm[enter_s2, 0] = 1.0
            self._gripper_cmd_pos[enter_s2, 0] = float(
                p["gripper_close_pos"]
            )

        # S2 is absorbing. Two consecutive lost-grasp HL steps terminate.
        stage2 = self._curriculum_stage == 2
        self._curriculum_s2_grasp_lost_count[:] = torch.where(
            stage2 & (~self._stage2),
            self._curriculum_s2_grasp_lost_count + 1,
            torch.zeros_like(self._curriculum_s2_grasp_lost_count),
        )
        self._curriculum_s2_grasp_lost_terminated[:] = (
            self._curriculum_s2_grasp_lost_count
            >= int(p["curriculum_s2_grasp_lost_consecutive_steps"])
        )

    def _update_retrieval_success(self) -> None:
        """Latch episode success after 3 consecutive Stage-2 retrieval hits."""
        p = self.cfg.env_params

        retrieval_error = torch.norm(
            self._object_pos_w - self._retrieval_target_pos_w,
            dim=-1,
        )
        success_now = (
            (self._curriculum_stage == 2)
            & (
                retrieval_error
                < float(p["retrieval_success_error_threshold"])
            )
        )

        self._retrieval_success_count[:] = torch.where(
            success_now & (~self._episode_retrieval_success),
            self._retrieval_success_count + 1,
            torch.where(
                self._episode_retrieval_success,
                self._retrieval_success_count,
                torch.zeros_like(self._retrieval_success_count),
            ),
        )

        newly_successful = (
            (~self._episode_retrieval_success)
            & (
                self._retrieval_success_count
                >= int(p["retrieval_success_consecutive_steps"])
            )
        )
        self._episode_retrieval_success[newly_successful] = True

    def _update_dynamic_reward_weights(
        self,
        env_ids: torch.Tensor,
        final_stage: torch.Tensor,
        episode_success: torch.Tensor,
        valid_episode: torch.Tensor,
    ) -> None:
        """Update weights from final stage; freeze after consecutive successful episodes."""
        if not torch.any(valid_episode):
            return

        valid_ids = env_ids[valid_episode]
        stages = final_stage[valid_episode]
        successful = episode_success[valid_episode]

        old_streak = self._curriculum_complete_episode_streak[valid_ids]
        new_streak = torch.where(
            successful,
            old_streak + 1,
            torch.zeros_like(old_streak),
        )
        self._curriculum_complete_episode_streak[valid_ids] = new_streak

        freeze_threshold = int(
            self.cfg.env_params[
                "dynamic_reward_freeze_after_consecutive_success_episodes"
            ]
        )
        newly_frozen = new_streak >= freeze_threshold
        self._curriculum_reward_scales_frozen[valid_ids[newly_frozen]] = True

        # The threshold-th successful episode triggers freezing; that reset does
        # not modify the scales. Environments already frozen stay frozen forever.
        can_update = ~self._curriculum_reward_scales_frozen[valid_ids]
        if not torch.any(can_update):
            return

        update_ids = valid_ids[can_update]
        update_stages = stages[can_update]
        multiplier = float(self.cfg.env_params["dynamic_reward_scale_multiplier"])
        if multiplier <= 1.0:
            raise ValueError("dynamic_reward_scale_multiplier must be > 1.")

        old = self._curriculum_reward_weights[update_ids]

        # Standard stage-owned curriculum rewards keep the original behavior:
        # the reward group matching the final stage is multiplied, while all other
        # standard reward groups are divided by the same multiplier.
        relevant = (
            self._curriculum_reward_stage.unsqueeze(0)
            == update_stages.unsqueeze(1)
        )
        updated = torch.where(relevant, old * multiplier, old / multiplier)

        # active_perception is NOT a normal single-stage curriculum reward.
        #
        # Desired semantics:
        #   final S0 -> unchanged
        #   final S1 -> unchanged
        #   final S2 -> anneal by / multiplier
        #
        # This makes it a shared S0/S1 auxiliary shaping reward whose importance
        # only decreases after the episode demonstrates successful progression to S2.
        ap_idx = self._active_perception_reward_idx
        ap_old = old[:, ap_idx]
        ap_reached_s2 = update_stages == 2
        ap_updated = torch.where(
            ap_reached_s2,
            ap_old / multiplier,
            ap_old,
        )
        updated[:, ap_idx] = ap_updated

        updated = torch.maximum(
            updated, self._curriculum_reward_weight_min.unsqueeze(0)
        )
        updated = torch.minimum(
            updated, self._curriculum_reward_weight_max.unsqueeze(0)
        )
        self._curriculum_reward_weights[update_ids] = updated

    def _apply_action(self):
        # Not used. We override step() and apply low-level actions inside the inner loop.
        pass

    # Low-level observation and action
    def _build_ll_obs_frame(self) -> torch.Tensor:
        """Build one LL obs frame matching the frozen low-level policy input layout."""
        joint_pos_rel = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel

        joint_pos_leg = joint_pos_rel[:, self.leg_ids]
        joint_pos_arm = joint_pos_rel[:, self.arm_ids]

        joint_vel_leg = joint_vel[:, self.leg_ids]
        joint_vel_arm = joint_vel[:, self.arm_ids]
        joint_vel_wheel = joint_vel[:, self.wheel_ids]

        frame = torch.cat(
            [
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self._base_velocity_cmd,
                self._ee_kp_cmd_plb,
                joint_pos_leg,
                joint_pos_arm,
                joint_vel_leg,
                joint_vel_arm,
                joint_vel_wheel,
                self._ll_actions,
            ],
            dim=-1,
        )

        if frame.shape[-1] != self.ll_frame_dim:
            raise RuntimeError(
                f"LL obs frame dim mismatch: expected {self.ll_frame_dim}, got {frame.shape[-1]}"
            )

        return frame

    def _flatten_ll_obs_history_feature_major(self) -> torch.Tensor:
        """Flatten LL history in the same order as successful MuJoCo sim2sim."""
        h = self._ll_obs_history
        n = self.num_envs

        obs = torch.cat(
            [
                h[:, :, 0:3].reshape(n, -1),
                h[:, :, 3:6].reshape(n, -1),
                h[:, :, 6:9].reshape(n, -1),
                h[:, :, 9:18].reshape(n, -1),
                h[:, :, 18:30].reshape(n, -1),
                h[:, :, 30:36].reshape(n, -1),
                h[:, :, 36:48].reshape(n, -1),
                h[:, :, 48:54].reshape(n, -1),
                h[:, :, 54:58].reshape(n, -1),
                h[:, :, 58:80].reshape(n, -1),
            ],
            dim=-1,
        )

        expected_dim = self.cfg.env_params["ll_obs_history"] * self.ll_frame_dim
        if obs.shape[-1] != expected_dim:
            raise RuntimeError(
                f"LL stacked obs dim mismatch: expected {expected_dim}, got {obs.shape[-1]}"
            )

        return obs

    def _build_ll_obs(self) -> torch.Tensor:
        frame = self._build_ll_obs_frame()

        self._ll_obs_history[:, :-1] = self._ll_obs_history[:, 1:].clone()
        self._ll_obs_history[:, -1] = frame

        return self._flatten_ll_obs_history_feature_major()

    def _apply_ll_action(self, ll_action: torch.Tensor):
        n_leg = self.num_leg
        n_arm = self.num_arm
        n_wheel = self.num_wheel

        leg_action = ll_action[:, :n_leg]
        arm_action = ll_action[:, n_leg:n_leg + n_arm]
        wheel_action = ll_action[:, n_leg + n_arm:n_leg + n_arm + n_wheel]

        leg_target = self._robot.data.default_joint_pos[:, self.leg_ids] + 0.25 * leg_action
        arm_target = (
            self._robot.data.default_joint_pos[:, self.arm_ids]
            + 0.10 * arm_action
        )

        wheel_vel_target = 4.0 * wheel_action

        self._robot.set_joint_position_target(leg_target, joint_ids=self.leg_ids)

        self._robot.set_joint_position_target(arm_target, joint_ids=self.arm_ids)

        self._robot.set_joint_velocity_target(wheel_vel_target, joint_ids=self.wheel_ids)

        # Gripper is directly controlled by high-level policy.
        self._robot.set_joint_position_target(
            self._gripper_cmd_pos,
            joint_ids=self.gripper_ids,
        )

    # Core step
    def step(self, actions: torch.Tensor):
        actions = actions.to(self.device)

        self._pre_physics_step(actions)

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        ll_steps = self.cfg.env_params["ll_steps_per_hl_step"]
        ll_decimation = self.cfg.env_params["ll_decimation"]

        for _ in range(ll_steps):
            ll_obs = self._build_ll_obs()

            with torch.no_grad():
                ll_action = self._ll_policy(ll_obs)

            if ll_action.shape[-1] != self.ll_action_dim:
                raise RuntimeError(
                    f"LL action dim mismatch: expected {self.ll_action_dim}, got {ll_action.shape[-1]}"
                )

            self._ll_actions[:] = ll_action

            self._apply_ll_action(ll_action)

            for _ in range(ll_decimation):
                self._sim_step_counter += 1
                self.scene.write_data_to_sim()
                self.sim.step(render=False)

                if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                    self.sim.render()

                self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self._update_task_buffers()
        self._update_grasp_stage()
        self._update_curriculum_stage()
        self._update_retrieval_success()

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs

        self.reward_buf = self._get_rewards()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()
            self._update_task_buffers()

        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.obs_buf = self._get_observations()

        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )

    # Task buffers
    def _update_task_buffers(self):
        p = self.cfg.env_params
        self._object_pos_w[:] = self._ball.data.root_pos_w

        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w
        root_quat_inv = math_utils.quat_inv(root_quat_w)

        object_rel_root_w = self._object_pos_w - root_pos_w
        self._object_center_pos_base[:] = math_utils.quat_apply(
            root_quat_inv, object_rel_root_w
        )
        self._object_height[:] = (
            self._object_pos_w[:, 2]
            - self.scene.env_origins[:, 2]
            - float(p["ground_z"])
        )

        object_points_plb = transform_points_w_to_plb(
            root_pos_w=root_pos_w,
            root_quat_w=root_quat_w,
            points_w=self._object_pos_w.unsqueeze(1),
            ground_z=p["ground_z"],
        )
        self._object_pos_plb[:] = object_points_plb[:, 0]


        object_quat_w = self._ball.data.root_quat_w
        half_extent = float(p["object_half_extent"])
        local_top_offset = torch.tensor(
            [0.0, 0.0, half_extent],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        local_bottom_offset = -local_top_offset

        object_top_pos_w = self._object_pos_w + math_utils.quat_apply(
            object_quat_w,
            local_top_offset,
        )
        object_bottom_pos_w = self._object_pos_w + math_utils.quat_apply(
            object_quat_w,
            local_bottom_offset,
        )
        object_face_points_w = torch.stack(
            [object_bottom_pos_w, object_top_pos_w],
            dim=1,
        )
        object_face_rel_root_w = object_face_points_w - root_pos_w.unsqueeze(1)
        root_quat_inv_faces = root_quat_inv.unsqueeze(1).expand(-1, 2, -1).reshape(-1, 4)
        object_face_points_base = math_utils.quat_apply(
            root_quat_inv_faces,
            object_face_rel_root_w.reshape(-1, 3),
        ).reshape(self.num_envs, 2, 3)
        self._object_bottom_center_pos_base[:] = object_face_points_base[:, 0]
        self._object_top_center_pos_base[:] = object_face_points_base[:, 1]

        object_face_points_plb = transform_points_w_to_plb(
            root_pos_w=root_pos_w,
            root_quat_w=root_quat_w,
            points_w=object_face_points_w,
            ground_z=p["ground_z"],
        )
        self._object_bottom_center_pos_plb[:] = object_face_points_plb[:, 0]
        self._object_top_center_pos_plb[:] = object_face_points_plb[:, 1]

        gripper_stator_pos_w = self._robot.data.body_pos_w[:, self.gripper_center_body_id_scalar]
        gripper_stator_quat_w = self._robot.data.body_quat_w[:, self.gripper_center_body_id_scalar]
        self._gripper_stator_pos_w[:] = gripper_stator_pos_w
        gripper_center_offset_w = math_utils.quat_apply(
            gripper_stator_quat_w,
            self._gripper_center_offset_local.unsqueeze(0).repeat(self.num_envs, 1),
        )
        self._gripper_center_pos_w[:] = gripper_stator_pos_w + gripper_center_offset_w
        self._gripper_center_pos_base[:] = math_utils.quat_apply(
            root_quat_inv,
            self._gripper_center_pos_w - root_pos_w,
        )

        gripper_quat_base = math_utils.quat_mul(root_quat_inv, gripper_stator_quat_w)
        local_x = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        local_y = torch.tensor([0.0, 1.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        gripper_x_axis_base = math_utils.quat_apply(gripper_quat_base, local_x)
        gripper_y_axis_base = math_utils.quat_apply(gripper_quat_base, local_y)
        self._gripper_orientation_base[:] = torch.cat(
            [gripper_x_axis_base, gripper_y_axis_base], dim=-1
        )

        object_rel_gripper_center_w = self._object_pos_w - self._gripper_center_pos_w
        self._object_pos_grippercenter[:] = math_utils.quat_apply(
            math_utils.quat_inv(gripper_stator_quat_w),
            object_rel_gripper_center_w,
        )
        self._grasp_error[:] = torch.norm(self._object_pos_grippercenter, dim=-1)

        gripper_center_plb = transform_points_w_to_plb(
            root_pos_w=root_pos_w,
            root_quat_w=root_quat_w,
            points_w=self._gripper_center_pos_w.unsqueeze(1),
            ground_z=p["ground_z"],
        )
        self._gripper_center_pos_plb[:] = gripper_center_plb[:, 0]
        self._object_grippercenter_error_plb[:] = (
            self._object_pos_plb - self._gripper_center_pos_plb
        )

        # Persistent retrieval target is stored in world frame; expose it in the
        # current full root/body frame so the policy gets a robot-centric goal.
        self._retrieval_target_pos_base[:] = math_utils.quat_apply(
            root_quat_inv,
            self._retrieval_target_pos_w - root_pos_w,
        )
        self._update_contact_buffers()

    # Observations
    def _build_hl_obs_frames(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build one actor frame and one privileged critic frame without shifting history."""
        joint_pos_rel = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel
        joint_pos_leg = joint_pos_rel[:, self.leg_ids]
        joint_pos_arm = joint_pos_rel[:, self.arm_ids]
        joint_pos_gripper = joint_pos_rel[:, self.gripper_ids]
        joint_vel_arm = joint_vel[:, self.arm_ids]

        object_center_pos_base = self._object_center_pos_base
        gripper_center_pos_base = self._gripper_center_pos_base
        retrieval_target_pos_base = self._retrieval_target_pos_base
        if self.cfg.add_noise and not self.cfg.play_mode:
            object_center_pos_base = self._add_observation_noise(
                object_center_pos_base,
                float(self.cfg.env_params["object_pos_detection_noise"]),
            )
            gripper_center_pos_base = self._add_observation_noise(
                gripper_center_pos_base,
                float(self.cfg.env_params["gripper_center_pos_noise"]),
            )
            retrieval_target_pos_base = self._add_observation_noise(
                retrieval_target_pos_base,
                float(self.cfg.env_params["object_target_error_noise"]),
            )

        # Apply one common small random rotation to both gripper axes. This keeps
        # the 6-D representation geometrically valid: both axes remain unit length
        # and mutually orthogonal. The critic continues to receive the true axes.
        gripper_orientation_actor = self._gripper_orientation_base
        if self.cfg.add_noise and not self.cfg.play_mode:
            orientation_noise = (
                torch.randn(self.num_envs, 3, device=self.device)
                * float(self.cfg.env_params["gripper_orientation_noise_std"])
            )
            orientation_noise_quat = math_utils.quat_from_euler_xyz(
                orientation_noise[:, 0],
                orientation_noise[:, 1],
                orientation_noise[:, 2],
            )
            gripper_x_axis_actor = math_utils.quat_apply(
                orientation_noise_quat,
                self._gripper_orientation_base[:, 0:3],
            )
            gripper_y_axis_actor = math_utils.quat_apply(
                orientation_noise_quat,
                self._gripper_orientation_base[:, 3:6],
            )
            gripper_orientation_actor = torch.cat(
                [gripper_x_axis_actor, gripper_y_axis_actor],
                dim=-1,
            )

        # Standard last-action semantics. The action stored in self._hl_actions is
        # the action that generated the just-completed transition. Base entries are
        # replaced by the effective normalized commands actually sent to the LL
        # controller after gate/clamp and grasp freeze.
        effective_base_action = torch.clamp(
            self._base_velocity_cmd / self._base_cmd_scale.unsqueeze(0),
            min=-1.0,
            max=1.0,
        )
        previous_hl_action = torch.cat(
            [
                effective_base_action,
                self._hl_actions[:, 3:8],
                self._gripper_cmd_norm,
            ],
            dim=-1,
        )

        actor_obs_frame = torch.cat([
            self._add_observation_noise(self._robot.data.root_ang_vel_b, 0.15),
            self._add_observation_noise(self._robot.data.projected_gravity_b, 0.05),
            self._add_observation_noise(joint_pos_leg, 0.01),
            self._add_observation_noise(joint_pos_arm, 0.01),
            self._add_observation_noise(joint_pos_gripper, 0.01),
            self._add_observation_noise(joint_vel_arm, 0.50),
            object_center_pos_base,
            gripper_orientation_actor,
            gripper_center_pos_base,
            retrieval_target_pos_base,
            previous_hl_action,
            self._grasp_confidence_proxy.float().unsqueeze(-1),
        ], dim=-1)

        critic_base_frame = torch.cat([
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
            joint_pos_leg,
            joint_pos_arm,
            joint_pos_gripper,
            joint_vel_arm,
            self._object_center_pos_base,
            self._gripper_orientation_base,
            self._gripper_center_pos_base,
            self._retrieval_target_pos_base,
            previous_hl_action,
            self._grasp_confidence_proxy.float().unsqueeze(-1),
        ], dim=-1)

        base_height = (
            self._robot.data.root_pos_w[:, 2]
            - self.scene.env_origins[:, 2]
            - float(self.cfg.env_params["ground_z"])
        ).unsqueeze(-1)
        object_mass = self._ball.root_physx_view.get_masses()
        object_mass = torch.as_tensor(
            object_mass, dtype=torch.float32, device=self.device
        ).reshape(self.num_envs, -1).sum(dim=-1, keepdim=True)
        object_side_length = torch.full(
            (self.num_envs, 1),
            float(self.cfg.env_params["cube_side"]),
            dtype=torch.float32,
            device=self.device,
        )
        privileged_extra = torch.cat([
            self._object_bottom_center_pos_base,
            self._object_top_center_pos_base,
            base_height,
            self._gripper_stator_cube_contact.float().unsqueeze(-1),
            self._gripper_mover_cube_contact.float().unsqueeze(-1),
            object_mass,
            object_side_length,
        ], dim=-1)
        critic_obs_frame = torch.cat([critic_base_frame, privileged_extra], dim=-1)

        if actor_obs_frame.shape[-1] != self.cfg.obs_dim:
            raise RuntimeError(
                f"HL actor obs dim mismatch: expected {self.cfg.obs_dim}, got {actor_obs_frame.shape[-1]}"
            )
        if critic_obs_frame.shape[-1] != self.cfg.critic_obs_dim:
            raise RuntimeError(
                f"Critic obs dim mismatch: expected {self.cfg.critic_obs_dim}, got {critic_obs_frame.shape[-1]}"
            )
        return actor_obs_frame, critic_obs_frame

    @staticmethod
    def _flatten_history_feature_major(
        history: torch.Tensor,
        feature_slices: tuple[tuple[int, int], ...],
    ) -> torch.Tensor:
        """Flatten [env, history, frame] by feature group, then old-to-new time."""
        n = history.shape[0]
        return torch.cat(
            [history[:, :, start:end].reshape(n, -1) for start, end in feature_slices],
            dim=-1,
        )

    def _flatten_actor_history_feature_major(self) -> torch.Tensor:
        # 56-D frame groups:
        # root ang vel, gravity, leg q, arm q, gripper q, arm qd, object pos,
        # gripper orientation, gripper center, retrieval target pos, last action,
        # grasp proxy. Root linear velocity is intentionally excluded from actor obs.
        slices = (
            (0, 3), (3, 6), (6, 18), (18, 24), (24, 25),
            (25, 31), (31, 34), (34, 40), (40, 43), (43, 46),
            (46, 55), (55, 56),
        )
        return self._flatten_history_feature_major(self._obs_history, slices)

    def _flatten_critic_history_feature_major(self) -> torch.Tensor:
        # Critic intentionally retains the original 59-D base layout, including
        # root linear velocity, followed by 11 privileged dims.
        slices = (
            (0, 3), (3, 6), (6, 9), (9, 21), (21, 27), (27, 28),
            (28, 34), (34, 37), (37, 43), (43, 46), (46, 49),
            (49, 58), (58, 59),
            (59, 62), (62, 65), (65, 66), (66, 67), (67, 68),
            (68, 69), (69, 70),
        )
        return self._flatten_history_feature_major(self._critic_history, slices)

    def _get_observations(self) -> dict:
        actor_obs_frame, critic_obs_frame = self._build_hl_obs_frames()
        self._obs_history[:, :-1] = self._obs_history[:, 1:].clone()
        self._obs_history[:, -1] = actor_obs_frame
        self._critic_history[:, :-1] = self._critic_history[:, 1:].clone()
        self._critic_history[:, -1] = critic_obs_frame

        policy_obs = self._flatten_actor_history_feature_major()
        critic_obs = self._flatten_critic_history_feature_major()
        expected_policy_dim = self.cfg.obs_dim * self.cfg.env_params["obs_history"]
        expected_critic_dim = self.cfg.critic_obs_dim * self.cfg.env_params["critic_obs_history"]
        if policy_obs.shape[-1] != expected_policy_dim:
            raise RuntimeError(
                f"HL actor stacked obs dim mismatch: expected {expected_policy_dim}, got {policy_obs.shape[-1]}"
            )
        if critic_obs.shape[-1] != expected_critic_dim:
            raise RuntimeError(
                f"HL critic stacked obs dim mismatch: expected {expected_critic_dim}, got {critic_obs.shape[-1]}"
            )
        return {"policy": policy_obs, "critic": critic_obs}

    def _update_contact_buffers(self):
        p = self.cfg.env_params

        net_forces = self._contact_sensor.data.net_forces_w
        base_forces = net_forces[:, self._base_contact_body_ids, :]
        self._base_contact[:] = torch.any(
            torch.norm(base_forces, dim=-1)
            > float(p["contact_force_threshold"]),
            dim=-1,
        )

        stator_force_matrix = (
            self._gripper_stator_cube_contact_sensor.data.force_matrix_w
        )
        mover_force_matrix = (
            self._gripper_mover_cube_contact_sensor.data.force_matrix_w
        )
        if stator_force_matrix is None or mover_force_matrix is None:
            raise RuntimeError(
                "Filtered gripper-cube contact sensors require force_matrix_w."
            )

        threshold = float(p["gripper_cube_contact_force_threshold"])
        self._gripper_stator_cube_contact[:] = torch.any(
            torch.norm(stator_force_matrix, dim=-1) > threshold,
            dim=(1, 2),
        )
        self._gripper_mover_cube_contact[:] = torch.any(
            torch.norm(mover_force_matrix, dim=-1) > threshold,
            dim=(1, 2),
        )

        # Static terrain filtering is unsupported by the PhysX GPU tensor API.
        # Detect likely ground contact from the generic robot contact sensor and
        # a world-height guard. The force test detects contact; the height test
        # prevents normal cube contact from being classified as ground contact.
        ground_force_threshold = float(
            p["gripper_ground_contact_force_threshold"]
        )
        ground_height_limit = (
            self.scene.env_origins[:, 2]
            + float(p["ground_z"])
            + float(p["gripper_ground_contact_height_threshold"])
        )

        stator_contact_force = torch.norm(
            net_forces[:, self._gripper_stator_contact_body_id, :],
            dim=-1,
        )
        mover_contact_force = torch.norm(
            net_forces[:, self._gripper_mover_contact_body_id, :],
            dim=-1,
        )
        stator_height = self._robot.data.body_pos_w[
            :, self._gripper_stator_body_id_scalar, 2
        ]
        mover_height = self._robot.data.body_pos_w[
            :, self._gripper_mover_body_id_scalar, 2
        ]

        self._gripper_stator_ground_contact[:] = (
            (stator_contact_force > ground_force_threshold)
            & (stator_height <= ground_height_limit)
        )
        self._gripper_mover_ground_contact[:] = (
            (mover_contact_force > ground_force_threshold)
            & (mover_height <= ground_height_limit)
        )
        self._gripper_ground_contact[:] = (
            self._gripper_stator_ground_contact
            | self._gripper_mover_ground_contact
        )

    # Rewards
    def _prepare_rewards(self):
        reward_names = [
            "grasp_align_coarse",
            "grasp_align_fine",
            "grasp_success_bonus",
            "active_perception",
            "gripper_ground_contact_penalty",
            "retrieval_target",
            "retrieval_heading",
            "stage2_gripper_close_action",
            "base_action_rate_penalty",
            "base_action_second_order_penalty",
            "base_velocity_command_abs_penalty",
            "end_effector_lin_vel_penalty",
            "end_effector_ang_vel_penalty",
            "gripper_open_cube_planar_velocity_penalty",
            "gripper_action_target_reward",
            "task_reward",
        ]
        self._episode_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in reward_names
        }
        metric_names = [
            "grasp_error",
            "grasp_xy_error",
            "grasp_pocket_object_local_x",
            "grasp_pocket_object_local_y",
            "grasp_pocket_object_local_z",
            "grasp_pocket_error_x",
            "grasp_pocket_error_y",
            "grasp_pocket_error_z",
            "grasp_pocket_l1_error",
            "grasp_pocket_3d_error",
            "grasp_align_coarse_kernel",
            "grasp_align_fine_kernel",
            "curriculum_stage",
            "curriculum_stage_s0",
            "curriculum_stage_s1",
            "curriculum_stage_s2",
            "curriculum_s0_to_s1_transition",
            "curriculum_s1_to_s0_transition",
            "curriculum_s1_to_s2_transition",
            "curriculum_s2_grasp_lost_count",
            "retrieval_success_count",
            "episode_retrieval_success",
            "curriculum_complete_episode_streak",
            "curriculum_reward_scales_frozen",
            "curriculum_weight_grasp_align_coarse",
            "curriculum_weight_grasp_align_fine",
            "curriculum_weight_gripper_action_target",
            "curriculum_weight_grasp_success_bonus",
            "curriculum_weight_active_perception",
            "curriculum_weight_retrieval_target",
            "curriculum_weight_retrieval_heading",
            "curriculum_weight_stage2_gripper_close_action",
            "stage2_gripper_close_action_factor",
            "grasp_confidence_proxy",
            "simulation_grasp_confidence",
            "grasp_just_confirmed",
            "grasp_bonus_awarded",
            "gripper_ground_contact",
            "gripper_stator_ground_contact",
            "gripper_mover_ground_contact",
            "active_perception_angle",
            "active_perception_cosine",
            "active_perception_kernel",
            "active_perception_camera_object_distance",
            "retrieval_target_error",
            "retrieval_target_kernel",
            "retrieval_heading_angle",
            "retrieval_heading_cosine",
            "retrieval_heading_kernel",
            "retrieval_target_pos_base_x",
            "retrieval_target_pos_base_y",
            "retrieval_target_pos_base_z",
            "base_action_rate",
            "base_action_second_order",
            "base_velocity_command_abs",
            "end_effector_lin_speed",
            "end_effector_ang_speed",
            "end_effector_lin_vel_squared",
            "end_effector_ang_vel_squared",
            "cube_planar_speed",
            "gripper_open_cube_planar_speed_after_deadband",
            "gripper_not_fully_closed",
            "gripper_close_commanded",
            "gripper_action_target",
            "gripper_action_target_error",
            "gripper_angle_delta",
            "gripper_angle_holding",
            "grasp_true_candidate",
            "grasp_true",
        ]
        self._episode_metrics = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in metric_names
        }

    def _get_rewards(self) -> torch.Tensor:
        p = self.cfg.env_params
        close_commanded = (self._gripper_cmd_norm[:, 0] > 0.0).float()

        coarse_weight = self._curriculum_reward_weights[:, 0]
        fine_weight = self._curriculum_reward_weights[:, 1]
        gripper_target_weight = self._curriculum_reward_weights[:, 2]
        grasp_bonus_weight = self._curriculum_reward_weights[:, 3]
        retrieval_target_weight = self._curriculum_reward_weights[:, 4]
        retrieval_heading_weight = self._curriculum_reward_weights[:, 5]
        stage2_gripper_close_action_weight = self._curriculum_reward_weights[:, 6]
        active_perception_weight = self._curriculum_reward_weights[:, 7]

        grasp_error = torch.norm(
            self._object_grippercenter_error_plb, dim=-1
        )
        grasp_xy_error = torch.norm(
            self._object_grippercenter_error_plb[:, :2], dim=-1
        )

        # Grasp-pocket alignment. Express the object center in the measured
        # gripperStator local frame:
        #   p_obj_g = R_g^T (p_obj_w - p_stator_w)
        # and track p_target_g = [0.075, 0.0, 0.0].
        gripper_stator_quat_w = self._robot.data.body_quat_w[
            :, self.ee_body_id_scalar
        ]
        object_pos_gripper_stator = math_utils.quat_apply(
            math_utils.quat_inv(gripper_stator_quat_w),
            self._object_pos_w - self._gripper_stator_pos_w,
        )
        # Use the single canonical gripper-center offset for grasp alignment too.
        # No second grasp-pocket target is maintained.
        grasp_pocket_axis_error = torch.abs(
            object_pos_gripper_stator - self._gripper_center_offset_local.unsqueeze(0)
        )
        grasp_pocket_error_x = grasp_pocket_axis_error[:, 0]
        grasp_pocket_error_y = grasp_pocket_axis_error[:, 1]
        grasp_pocket_error_z = grasp_pocket_axis_error[:, 2]
        grasp_pocket_l1_error = torch.sum(grasp_pocket_axis_error, dim=-1)

        grasp_pocket_3d_error = torch.norm(
            grasp_pocket_axis_error,
            dim=-1,
        )
        grasp_align_coarse_kernel = self._exp_abs_reward(
            grasp_pocket_3d_error,
            float(p["grasp_align_coarse_exp_std"]),
        )

        grasp_align_fine_kernel = self._exp_abs_reward(
            grasp_pocket_3d_error,
            float(p["grasp_align_fine_exp_std"]),
        )

        grasp_align_coarse = coarse_weight * grasp_align_coarse_kernel
        grasp_align_fine = fine_weight * grasp_align_fine_kernel

        gripper_ground_contact_penalty = -(
            float(p["gripper_ground_contact_penalty_weight"])
            * self._gripper_ground_contact.float()
        )

        # Pre-grasp active-perception objective (grasp-proxy gated).
        #
        # Definition:
        #   minimize the full 3-D angle between
        #       robot heading (base_link local +X axis)
        #   and
        #       front-camera-center -> object-center line of sight.
        #
        # The camera center uses the exact f_oc_link translation from b2wz1.urdf:
        #   xyz = [0.3993, 0.0, -0.01576] in base_link.
        #
        # Note that camera optical-axis orientation is intentionally irrelevant here:
        # the camera is only the physical origin of the object line of sight.
        base_link_pos_w = self._robot.data.body_pos_w[
            :, self.base_link_body_id_scalar
        ]
        base_link_quat_w = self._robot.data.body_quat_w[
            :, self.base_link_body_id_scalar
        ]

        camera_pos_base = self._front_optical_camera_pos_base.unsqueeze(0).expand(
            self.num_envs, -1
        )
        front_camera_pos_w = base_link_pos_w + math_utils.quat_apply(
            base_link_quat_w,
            camera_pos_base,
        )

        # Robot heading is the base_link local +X axis expressed in world frame.
        robot_heading_w = math_utils.quat_apply(
            base_link_quat_w,
            torch.tensor(
                [1.0, 0.0, 0.0],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0).expand(self.num_envs, -1),
        )
        robot_heading_w = robot_heading_w / torch.clamp(
            torch.norm(robot_heading_w, dim=-1, keepdim=True),
            min=1.0e-6,
        )

        # Full 3-D front-camera-center -> object-center line of sight.
        camera_to_object_w = self._object_pos_w - front_camera_pos_w
        camera_object_distance = torch.norm(
            camera_to_object_w,
            dim=-1,
            keepdim=True,
        )
        camera_to_object_unit_w = camera_to_object_w / torch.clamp(
            camera_object_distance,
            min=1.0e-6,
        )

        active_perception_cosine = torch.sum(
            robot_heading_w * camera_to_object_unit_w,
            dim=-1,
        ).clamp(-1.0, 1.0)
        active_perception_angle = torch.acos(active_perception_cosine)

        # If camera center and object center coincide numerically, LOS is undefined.
        # Treat that degenerate case as aligned to avoid an arbitrary orientation term.
        perception_degenerate = camera_object_distance[:, 0] < 1.0e-4
        active_perception_angle = torch.where(
            perception_degenerate,
            torch.zeros_like(active_perception_angle),
            active_perception_angle,
        )
        active_perception_cosine = torch.where(
            perception_degenerate,
            torch.ones_like(active_perception_cosine),
            active_perception_cosine,
        )

        active_perception_kernel = self._exp_abs_reward(
            active_perception_angle,
            float(p["active_perception_exp_std"]),
        )
        # Deployable grasp-proxy gating:
        # active perception is used only before the grasp proxy is confirmed.
        #
        #   grasp_confidence_proxy == 0 -> active
        #   grasp_confidence_proxy == 1 -> zero reward
        #
        # This intentionally does not gate on the simulation-only curriculum stage.
        # Therefore, if the deployable proxy becomes true while the environment is
        # still in S1 waiting for dual-contact confirmation, active perception is
        # already disabled, matching deployment semantics.
        active_perception_active = (~self._grasp_confidence_proxy).float()
        active_perception_reward = (
            active_perception_active
            * active_perception_weight
            * active_perception_kernel
        )

        # Stage-2 retrieval objective. Euclidean 3-D error is frame-invariant;
        # compute it directly in world frame against the persistent sampled goal.
        retrieval_target_error = torch.norm(
            self._object_pos_w - self._retrieval_target_pos_w,
            dim=-1,
        )
        retrieval_target_kernel = self._exp_abs_reward(
            retrieval_target_error,
            float(p["retrieval_target_exp_std"]),
        )
        simulation_grasp_confidence = self._stage2.float()
        retrieval_target_reward = (
            simulation_grasp_confidence
            * retrieval_target_weight
            * retrieval_target_kernel
        )

        # Stage-2 heading objective.
        # Compute the root pose and planar +X heading locally in this reward
        # function. These variables must not depend on the Stage-1
        # active-perception block above.
        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w

        base_x_axis_w = math_utils.quat_apply(
            root_quat_w,
            torch.tensor(
                [1.0, 0.0, 0.0],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0).repeat(self.num_envs, 1),
        )
        base_heading_xy = base_x_axis_w[:, :2]
        base_heading_xy = base_heading_xy / torch.clamp(
            torch.norm(base_heading_xy, dim=-1, keepdim=True),
            min=1.0e-6,
        )

        # Compare the robot root/base +X heading with the current
        # base->retrieval-target direction in world XY.
        target_direction_xy = (
            self._retrieval_target_pos_w[:, :2] - root_pos_w[:, :2]
        )
        target_distance_xy = torch.norm(
            target_direction_xy,
            dim=-1,
            keepdim=True,
        )
        target_direction_xy_unit = target_direction_xy / torch.clamp(
            target_distance_xy,
            min=1.0e-6,
        )

        retrieval_heading_cosine = torch.sum(
            base_heading_xy * target_direction_xy_unit,
            dim=-1,
        ).clamp(-1.0, 1.0)
        retrieval_heading_angle = torch.acos(retrieval_heading_cosine)

        # At the exact target XY, heading is undefined. Treat it as aligned so the
        # reward does not inject an arbitrary orientation objective at zero distance.
        target_xy_degenerate = target_distance_xy[:, 0] < 1.0e-4
        retrieval_heading_angle = torch.where(
            target_xy_degenerate,
            torch.zeros_like(retrieval_heading_angle),
            retrieval_heading_angle,
        )
        retrieval_heading_cosine = torch.where(
            target_xy_degenerate,
            torch.ones_like(retrieval_heading_cosine),
            retrieval_heading_cosine,
        )

        retrieval_heading_kernel = self._exp_abs_reward(
            retrieval_heading_angle,
            float(p["retrieval_heading_exp_std"]),
        )
        retrieval_heading_reward = (
            simulation_grasp_confidence
            * retrieval_heading_weight
            * retrieval_heading_kernel
        )

        # Stage-2 policy gripper-close objective.
        # Use the raw policy action through _gripper_close_confidence rather than
        # the executed binary command.  Therefore the policy receives the largest
        # reward at raw gripper action +1, even if the Stage-2 heuristic is already
        # forcing the physical gripper command closed.
        stage2_gripper_close_action_factor = torch.clamp(
            self._gripper_close_confidence,
            min=0.0,
            max=1.0,
        )
        stage2_gripper_close_action_reward = (
            simulation_grasp_confidence
            * stage2_gripper_close_action_weight
            * stage2_gripper_close_action_factor
        )

        # One bonus per episode, scaled dynamically as an S1 reward.
        grasp_just_confirmed = (
            self._curriculum_s1_to_s2_event
            & (~self._grasp_bonus_awarded)
        )
        grasp_success_bonus = (
            grasp_bonus_weight * grasp_just_confirmed.float()
        )
        self._grasp_bonus_awarded[grasp_just_confirmed] = True

        base_action_delta = (
            self._hl_actions[:, 0:3] - self._prev_hl_actions[:, 0:3]
        )
        base_action_rate = torch.sum(torch.square(base_action_delta), dim=-1)
        base_action_second_order = torch.sum(
            torch.square(
                self._hl_actions[:, :3]
                - 2.0 * self._prev_hl_actions[:, :3]
                + self._prev_prev_hl_actions[:, :3]
            ),
            dim=-1,
        )
        # Penalties are fixed and never curriculum-scaled.
        base_action_rate_penalty = -(
            float(p["base_action_rate_penalty_weight"]) * base_action_rate
        )
        base_action_second_order_penalty = -(
            float(p["base_action_second_order_penalty_weight"])
            * base_action_second_order
        )
        base_velocity_command_abs = torch.sum(
            torch.abs(self._base_velocity_cmd), dim=-1
        )
        base_velocity_command_abs_penalty = -(
            float(p["base_velocity_command_abs_penalty_weight"])
            * base_velocity_command_abs
        )
        end_effector_lin_vel_w = self._robot.data.body_lin_vel_w[
            :, self.ee_body_id_scalar
        ]
        end_effector_ang_vel_w = self._robot.data.body_ang_vel_w[
            :, self.ee_body_id_scalar
        ]
        end_effector_lin_vel_squared = torch.sum(
            torch.square(end_effector_lin_vel_w), dim=-1
        )
        end_effector_ang_vel_squared = torch.sum(
            torch.square(end_effector_ang_vel_w), dim=-1
        )
        end_effector_lin_speed = torch.sqrt(
            torch.clamp(end_effector_lin_vel_squared, min=0.0)
        )
        end_effector_ang_speed = torch.sqrt(
            torch.clamp(end_effector_ang_vel_squared, min=0.0)
        )
        end_effector_lin_vel_penalty = -(
            float(p["end_effector_lin_vel_penalty_weight"])
            * end_effector_lin_vel_squared
        )
        end_effector_ang_vel_penalty = -(
            float(p["end_effector_ang_vel_penalty_weight"])
            * end_effector_ang_vel_squared
        )

        cube_planar_speed = torch.norm(
            self._ball.data.root_lin_vel_w[:, :2], dim=-1
        )
        gripper_open = self._gripper_cmd_norm[:, 0] < 0.0
        gripper_open_cube_planar_speed_after_deadband = torch.clamp(
            cube_planar_speed
            - float(p["gripper_open_cube_planar_velocity_deadband"]),
            min=0.0,
        )
        gripper_open_cube_planar_velocity_penalty = -(
            float(p["gripper_open_cube_planar_velocity_penalty_weight"])
            * gripper_open.float()
            * torch.square(gripper_open_cube_planar_speed_after_deadband)
        )

        gripper_near = (
            grasp_error
            <= float(p["gripper_action_target_error_threshold"])
        )
        gripper_action_target = gripper_near.float()
        gripper_action_target_error = torch.abs(
            self._gripper_close_confidence - gripper_action_target
        )

        # Gripper timing shaping:
        #   far  + open  -> 0
        #   far  + close -> negative
        #   near + open  -> 0
        #   near + close -> positive
        #
        # The physical gripper command is binary at raw action 0.  For shaping,
        # use only the positive side of the raw action as continuous close intent.
        # Therefore every raw action < 0 (executed OPEN) receives exactly zero
        # gripper-action reward instead of being rewarded for staying open.
        gripper_close_intent = torch.clamp(
            self._raw_gripper_action,
            min=0.0,
            max=1.0,
        )
        gripper_action_reward_sign = torch.where(
            gripper_near,
            torch.ones_like(gripper_close_intent),
            -torch.ones_like(gripper_close_intent),
        )
        gripper_action_target_reward = (
            gripper_target_weight
            * gripper_action_reward_sign
            * gripper_close_intent
        )

        reward = (
            grasp_align_coarse
            + grasp_align_fine
            + grasp_success_bonus
            + active_perception_reward
            + gripper_ground_contact_penalty
            + retrieval_target_reward
            + retrieval_heading_reward
            + stage2_gripper_close_action_reward
            + base_action_rate_penalty
            + base_action_second_order_penalty
            + base_velocity_command_abs_penalty
            + end_effector_lin_vel_penalty
            + end_effector_ang_vel_penalty
            + gripper_open_cube_planar_velocity_penalty
            + gripper_action_target_reward
        )
        if not torch.isfinite(reward).all():
            raise RuntimeError("Non-finite grasp-task reward detected.")

        reward_values = {
            "grasp_align_coarse": grasp_align_coarse,
            "grasp_align_fine": grasp_align_fine,
            "grasp_success_bonus": grasp_success_bonus,
            "active_perception": active_perception_reward,
            "gripper_ground_contact_penalty": gripper_ground_contact_penalty,
            "retrieval_target": retrieval_target_reward,
            "retrieval_heading": retrieval_heading_reward,
            "stage2_gripper_close_action": stage2_gripper_close_action_reward,
            "base_action_rate_penalty": base_action_rate_penalty,
            "base_action_second_order_penalty": base_action_second_order_penalty,
            "base_velocity_command_abs_penalty": base_velocity_command_abs_penalty,
            "end_effector_lin_vel_penalty": end_effector_lin_vel_penalty,
            "end_effector_ang_vel_penalty": end_effector_ang_vel_penalty,
            "gripper_open_cube_planar_velocity_penalty": gripper_open_cube_planar_velocity_penalty,
            "gripper_action_target_reward": gripper_action_target_reward,
            "task_reward": reward,
        }
        for name, value in reward_values.items():
            self._episode_sums[name] += value

        metric_values = {
            "grasp_error": grasp_error,
            "grasp_xy_error": grasp_xy_error,
            "grasp_pocket_object_local_x": object_pos_gripper_stator[:, 0],
            "grasp_pocket_object_local_y": object_pos_gripper_stator[:, 1],
            "grasp_pocket_object_local_z": object_pos_gripper_stator[:, 2],
            "grasp_pocket_error_x": grasp_pocket_error_x,
            "grasp_pocket_error_y": grasp_pocket_error_y,
            "grasp_pocket_error_z": grasp_pocket_error_z,
            "grasp_pocket_l1_error": grasp_pocket_l1_error,
            "grasp_pocket_3d_error": grasp_pocket_3d_error,
            "grasp_align_coarse_kernel": grasp_align_coarse_kernel,
            "grasp_align_fine_kernel": grasp_align_fine_kernel,
            "curriculum_stage": self._curriculum_stage.float(),
            "curriculum_stage_s0": (self._curriculum_stage == 0).float(),
            "curriculum_stage_s1": (self._curriculum_stage == 1).float(),
            "curriculum_stage_s2": (self._curriculum_stage == 2).float(),
            "curriculum_s0_to_s1_transition": self._curriculum_s0_to_s1_event.float(),
            "curriculum_s1_to_s0_transition": self._curriculum_s1_to_s0_event.float(),
            "curriculum_s1_to_s2_transition": self._curriculum_s1_to_s2_event.float(),
            "curriculum_s2_grasp_lost_count": self._curriculum_s2_grasp_lost_count.float(),
            "retrieval_success_count": self._retrieval_success_count.float(),
            "episode_retrieval_success": self._episode_retrieval_success.float(),
            "curriculum_complete_episode_streak": self._curriculum_complete_episode_streak.float(),
            "curriculum_reward_scales_frozen": self._curriculum_reward_scales_frozen.float(),
            "curriculum_weight_grasp_align_coarse": coarse_weight,
            "curriculum_weight_grasp_align_fine": fine_weight,
            "curriculum_weight_gripper_action_target": gripper_target_weight,
            "curriculum_weight_grasp_success_bonus": grasp_bonus_weight,
            "curriculum_weight_active_perception": active_perception_weight,
            "curriculum_weight_retrieval_target": retrieval_target_weight,
            "curriculum_weight_retrieval_heading": retrieval_heading_weight,
            "curriculum_weight_stage2_gripper_close_action": stage2_gripper_close_action_weight,
            "stage2_gripper_close_action_factor": stage2_gripper_close_action_factor,
            "grasp_confidence_proxy": self._grasp_confidence_proxy.float(),
            "simulation_grasp_confidence": simulation_grasp_confidence,
            "grasp_just_confirmed": grasp_just_confirmed.float(),
            "grasp_bonus_awarded": self._grasp_bonus_awarded.float(),
            "gripper_ground_contact": self._gripper_ground_contact.float(),
            "gripper_stator_ground_contact": self._gripper_stator_ground_contact.float(),
            "gripper_mover_ground_contact": self._gripper_mover_ground_contact.float(),
            "active_perception_angle": active_perception_angle,
            "active_perception_cosine": active_perception_cosine,
            "active_perception_kernel": active_perception_kernel,
            "active_perception_camera_object_distance": camera_object_distance[:, 0],
            "retrieval_target_error": retrieval_target_error,
            "retrieval_target_kernel": retrieval_target_kernel,
            "retrieval_heading_angle": retrieval_heading_angle,
            "retrieval_heading_cosine": retrieval_heading_cosine,
            "retrieval_heading_kernel": retrieval_heading_kernel,
            "retrieval_target_pos_base_x": self._retrieval_target_pos_base[:, 0],
            "retrieval_target_pos_base_y": self._retrieval_target_pos_base[:, 1],
            "retrieval_target_pos_base_z": self._retrieval_target_pos_base[:, 2],
            "base_action_rate": base_action_rate,
            "base_action_second_order": base_action_second_order,
            "base_velocity_command_abs": base_velocity_command_abs,
            "end_effector_lin_speed": end_effector_lin_speed,
            "end_effector_ang_speed": end_effector_ang_speed,
            "end_effector_lin_vel_squared": end_effector_lin_vel_squared,
            "end_effector_ang_vel_squared": end_effector_ang_vel_squared,
            "cube_planar_speed": cube_planar_speed,
            "gripper_open_cube_planar_speed_after_deadband": gripper_open_cube_planar_speed_after_deadband,
            "gripper_not_fully_closed": self._gripper_not_fully_closed.float(),
            "gripper_close_commanded": close_commanded,
            "gripper_action_target": gripper_action_target,
            "gripper_action_target_error": gripper_action_target_error,
            "gripper_angle_delta": self._gripper_angle_delta,
            "gripper_angle_holding": self._gripper_angle_holding.float(),
            "grasp_true_candidate": self._grasp_true_candidate.float(),
            "grasp_true": self._stage2.float(),
        }
        for name, value in metric_values.items():
            self._episode_metrics[name] += value

        self._prev_grasp_confidence_proxy[:] = self._grasp_confidence_proxy
        self._prev_stage2[:] = self._stage2
        return reward

    # Dones
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.cfg.env_params

        grasp_error_oob_now = (
            self._grasp_error
            > float(p["grasp_error_termination_threshold"])
        )
        self._grasp_error_oob_steps[:] = torch.where(
            grasp_error_oob_now,
            self._grasp_error_oob_steps + 1,
            torch.zeros_like(self._grasp_error_oob_steps),
        )
        self._grasp_error_oob_terminated[:] = (
            self._grasp_error_oob_steps
            >= int(p["grasp_error_termination_consecutive_steps"])
        )

        terminated = (
            self._base_contact
            | self._grasp_error_oob_terminated
            | self._curriculum_s2_grasp_lost_terminated
        )
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        elif len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        base_contact_reset_envs = self._base_contact[env_ids].clone()
        grasp_error_oob_reset_envs = (
            self._grasp_error_oob_terminated[env_ids].clone()
        )
        final_grasp_error_reset_envs = self._grasp_error[env_ids].clone()
        final_object_pos_plb_reset_envs = self._object_pos_plb[env_ids].clone()
        final_retrieval_target_error_reset_envs = torch.norm(
            self._object_pos_w[env_ids] - self._retrieval_target_pos_w[env_ids],
            dim=-1,
        ).clone()

        s2_grasp_lost_reset_envs = (
            self._curriculum_s2_grasp_lost_terminated[env_ids].clone()
        )
        final_curriculum_stage_reset_envs = self._curriculum_stage[env_ids].clone()
        episode_retrieval_success_reset_envs = (
            self._episode_retrieval_success[env_ids].clone()
        )
        final_curriculum_weights_reset_envs = (
            self._curriculum_reward_weights[env_ids].clone()
        )
        valid_curriculum_episode = self.episode_length_buf[env_ids] > 0
        self._update_dynamic_reward_weights(
            env_ids,
            final_curriculum_stage_reset_envs,
            episode_retrieval_success_reset_envs,
            valid_curriculum_episode,
        )

        self._robot.reset(env_ids)
        self._ball.reset(env_ids)

        average_step_reward = {}
        for key in self._episode_sums.keys():
            average_step_reward[key] = self._episode_sums[key][env_ids] / (
                self.episode_length_buf[env_ids] + 1.0e-6
            )

        average_step_metric = {}
        for key in self._episode_metrics.keys():
            average_step_metric[key] = self._episode_metrics[key][env_ids] / (
                self.episode_length_buf[env_ids] + 1.0e-6
            )

        super()._reset_idx(env_ids)

        self._grasp_error_oob_steps[env_ids] = 0
        self._grasp_error_oob_terminated[env_ids] = False
        self._grasp_confidence_proxy[env_ids] = False
        self._prev_grasp_confidence_proxy[env_ids] = False
        self._grasp_proxy_candidate[env_ids] = False
        self._grasp_proxy_enter_count[env_ids] = 0
        self._grasp_proxy_exit_count[env_ids] = 0
        self._stage2[env_ids] = False
        self._prev_stage2[env_ids] = False
        self._grasp_true_candidate[env_ids] = False

        self._curriculum_stage[env_ids] = 0
        self._curriculum_s0_to_s1_count[env_ids] = 0
        self._curriculum_s1_to_s0_count[env_ids] = 0
        self._curriculum_s2_grasp_lost_count[env_ids] = 0
        self._curriculum_s2_grasp_lost_terminated[env_ids] = False
        self._curriculum_s0_to_s1_event[env_ids] = False
        self._curriculum_s1_to_s0_event[env_ids] = False
        self._curriculum_s1_to_s2_event[env_ids] = False
        self._grasp_bonus_awarded[env_ids] = False
        self._retrieval_success_count[env_ids] = 0
        self._episode_retrieval_success[env_ids] = False

        if len(env_ids) == self.num_envs and not self.cfg.play_mode:
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf,
                high=int(self.max_episode_length),
            )

        self._ll_obs_history[env_ids] = 0.0
        self._ll_actions[env_ids] = 0.0

        self._hl_actions[env_ids] = 0.0
        self._prev_hl_actions[env_ids] = 0.0
        self._prev_prev_hl_actions[env_ids] = 0.0

        self._obs_history[env_ids] = 0.0
        self._critic_history[env_ids] = 0.0

        self._base_velocity_cmd[env_ids] = 0.0
        self._kp0_cmd[env_ids] = self._neutral_kp0
        self._prev_kp0_cmd[env_ids] = self._neutral_kp0
        self._ee_yaw_cmd[env_ids] = self.cfg.env_params["neutral_ee_yaw"]
        self._ee_pitch_cmd[env_ids] = self.cfg.env_params["neutral_ee_pitch"]
        self._ee_roll_cmd[env_ids] = self.cfg.env_params["fixed_ee_roll"]
        self._prev_ee_yaw_cmd[env_ids] = self._ee_yaw_cmd[env_ids]
        self._prev_ee_pitch_cmd[env_ids] = self._ee_pitch_cmd[env_ids]
        self._ee_kp_cmd_plb[env_ids] = 0.0

        self._gripper_cmd_norm[env_ids] = -1.0
        self._prev_gripper_cmd_norm[env_ids] = -1.0
        self._gripper_cmd_pos[env_ids] = self.cfg.env_params["gripper_open_pos"]
        self._raw_gripper_action[env_ids] = -1.0
        self._gripper_close_confidence[env_ids] = 0.0
        self._prev_gripper_joint_pos[env_ids] = float(
            self.cfg.env_params["gripper_open_pos"]
        )
        self._gripper_angle_delta[env_ids] = 0.0
        self._gripper_not_fully_closed[env_ids] = False
        self._gripper_angle_holding[env_ids] = False

        self._object_pos_w[env_ids] = 0.0
        self._object_pos_plb[env_ids] = 0.0
        self._object_center_pos_base[env_ids] = 0.0
        self._object_height[env_ids] = 0.0
        self._retrieval_target_pos_w[env_ids] = 0.0
        self._retrieval_target_pos_base[env_ids] = 0.0
        self._object_grippercenter_error_plb[env_ids] = 0.0
        self._object_bottom_center_pos_plb[env_ids] = 0.0
        self._object_top_center_pos_plb[env_ids] = 0.0
        self._object_bottom_center_pos_base[env_ids] = 0.0
        self._object_top_center_pos_base[env_ids] = 0.0
        self._object_pos_grippercenter[env_ids] = 0.0
        self._gripper_center_pos_plb[env_ids] = 0.0
        self._gripper_center_pos_base[env_ids] = 0.0
        self._gripper_orientation_base[env_ids] = 0.0
        self._gripper_center_pos_w[env_ids] = 0.0
        self._grasp_error[env_ids] = 0.0
        self._base_contact[env_ids] = False
        self._gripper_stator_cube_contact[env_ids] = False
        self._gripper_mover_cube_contact[env_ids] = False
        self._gripper_stator_ground_contact[env_ids] = False
        self._gripper_mover_ground_contact[env_ids] = False
        self._gripper_ground_contact[env_ids] = False

        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()

        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        if not self.cfg.play_mode:
            p = self.cfg.env_params
            pose_range = p["reset_base_pose_range"]
            velocity_range = p["reset_base_velocity_range"]
            n = len(env_ids)

            def sample_uniform(bounds: tuple[float, float]) -> torch.Tensor:
                return torch.empty(n, device=self.device).uniform_(float(bounds[0]), float(bounds[1]))

            default_root_state[:, 2] += sample_uniform(pose_range["z"])

            roll = sample_uniform(pose_range["roll"])
            pitch = sample_uniform(pose_range["pitch"])
            yaw = torch.zeros(n, device=self.device)
            random_quat = math_utils.quat_from_euler_xyz(
                roll,
                pitch,
                yaw,
            )
            default_root_state[:, 3:7] = math_utils.quat_mul(
                random_quat,
                default_root_state[:, 3:7],
            )

            velocity_keys = ("x", "y", "z", "roll", "pitch", "yaw")
            for i, key in enumerate(velocity_keys):
                default_root_state[:, 7 + i] += sample_uniform(
                    velocity_range[key]
                )

            leg_scale = torch.empty(
                n,
                1,
                device=self.device,
            ).uniform_(
                *p["reset_leg_joint_position_scale_range"]
            )
            joint_pos[:, self.leg_ids] *= leg_scale

            arm_scale = torch.empty(
                n,
                1,
                device=self.device,
            ).uniform_(
                *p["reset_arm_joint_position_scale_range"]
            )
            joint_pos[:, self.arm_ids] *= arm_scale

        # Reset gripper open so the policy starts from a valid pre-grasp state.
        joint_pos[:, self.gripper_ids] = self.cfg.env_params["gripper_open_pos"]
        joint_vel[:, self.gripper_ids] = 0.0

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Flush the robot reset first, then read the actual root pose.
        # Object reset is defined in PLB, the same frame used by the frozen
        # low-level keypoint command: origin=[root_x, root_y, ground_z],
        # orientation=yaw-only(root_quat).
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=0.0)

        root_pos_w = self._robot.data.root_pos_w[env_ids].clone()
        root_quat_w = self._robot.data.root_quat_w[env_ids].clone()

        # Sample one persistent retrieval goal per episode.
        #
        # XY is sampled uniformly BY AREA inside the world-frame disk centered at
        # the initialized robot root XY. sqrt(U) is required for a spatially
        # uniform disk distribution rather than concentrating samples near the center.
        target_theta = torch.empty(
            len(env_ids), device=self.device
        ).uniform_(-math.pi, math.pi)
        max_target_radius = float(self.cfg.env_params["retrieval_target_radius"])
        target_radius = max_target_radius * torch.sqrt(
            torch.rand(len(env_ids), device=self.device)
        )
        self._retrieval_target_pos_w[env_ids, 0] = (
            root_pos_w[:, 0] + target_radius * torch.cos(target_theta)
        )
        self._retrieval_target_pos_w[env_ids, 1] = (
            root_pos_w[:, 1] + target_radius * torch.sin(target_theta)
        )

        # Sample target world Z independently for every reset environment.
        target_z_range_w = self.cfg.env_params["retrieval_target_z_range_w"]
        target_z_min = float(target_z_range_w[0])
        target_z_max = float(target_z_range_w[1])
        if target_z_max < target_z_min:
            raise ValueError(
                "retrieval_target_z_range_w must satisfy z_max >= z_min."
            )
        self._retrieval_target_pos_w[env_ids, 2] = torch.empty(
            len(env_ids),
            dtype=torch.float32,
            device=self.device,
        ).uniform_(target_z_min, target_z_max)

        # The retrieval target is static in world frame for the whole episode,
        # so updating the marker once here is sufficient. Visualize all envs so
        # partial resets preserve markers for environments that did not reset.
        if bool(self.cfg.env_params["retrieval_target_marker_vis"]):
            self._retrieval_target_markers.visualize(
                translations=self._retrieval_target_pos_w
            )

        plb_pos_w = root_pos_w.clone()
        plb_pos_w[:, 2] = self.scene.env_origins[env_ids, 2] + self.cfg.env_params["ground_z"]
        plb_yaw_quat_w = math_utils.yaw_quat(root_quat_w)

        # Sample and spawn the real cube object in the PLB frame.  Use yaw-only
        # so reset roll/pitch do not tilt the horizontal object offset into the
        # ground/air.  The x/y offset is sampled from a pregrasp-centered polar
        # sector rather than a rectangular box:
        #   r = U(r_min, r_max), theta = U(-theta_max, theta_max)
        #   object_x = x_center + r*cos(theta)
        #   object_y = y_center + r*sin(theta)
        object_x, object_y = self._sample_object_xy_plb(len(env_ids))
        object_z = torch.full(
            (len(env_ids),),
            float(self.cfg.env_params["object_half_extent"])
            + float(self.cfg.env_params.get("object_spawn_z_epsilon", 0.0)),
            dtype=torch.float32,
            device=self.device,
        )
        object_rel_plb = torch.stack([object_x, object_y, object_z], dim=-1)
        object_pos_w = plb_pos_w + math_utils.quat_apply(plb_yaw_quat_w, object_rel_plb)
        # Do not manually seed _object_pos_w here.  It is a task buffer and must
        # be filled from the actual simulated RigidObject state in
        # _update_task_buffers(), otherwise markers/observations can disagree
        # with the real PhysX cube pose.

        ball_root_state = self._ball.data.default_root_state[env_ids].clone()
        ball_root_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0],
            dtype=torch.float32,
            device=self.device,
        ).repeat(len(env_ids), 1)
        ball_root_state[:, 3:7] = ball_root_quat
        ball_root_state[:, 7:13] = 0.0

        # IMPORTANT: in this IsaacLab setup, RigidObject.write_root_pose_to_sim()
        # expects WORLD-frame root poses.  Earlier env-local writes placed all
        # cubes near the world origin / wrong tiles, making them appear detached
        # from their robots.  Write the intended world pose directly, and never
        # manually overwrite _object_pos_w or self._ball.data.root_state_w below;
        # task buffers/markers must come from the actual simulated readback.
        intended_object_pos_w = object_pos_w.clone()

        ball_root_state[:, :3] = intended_object_pos_w
        self._ball.write_root_pose_to_sim(ball_root_state[:, :7], env_ids)
        self._ball.write_root_velocity_to_sim(ball_root_state[:, 7:13], env_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=0.0)

        # Read current task geometry directly from simulator state.
        self._update_task_buffers()

        self._prev_gripper_joint_pos[env_ids] = (
            self._robot.data.joint_pos[env_ids][:, self.gripper_ids[0]]
        )
        # Neutral low-level command.
        # Important:
        # This avoids filling LL history with zero EE command after reset.
        neutral_ee_kp_cmd = self._neutral_ee_keypoints_plb(num=len(env_ids))
        self._kp0_cmd[env_ids] = self._neutral_kp0
        self._prev_kp0_cmd[env_ids] = self._neutral_kp0
        self._ee_yaw_cmd[env_ids] = self.cfg.env_params["neutral_ee_yaw"]
        self._ee_pitch_cmd[env_ids] = self.cfg.env_params["neutral_ee_pitch"]
        self._ee_roll_cmd[env_ids] = self.cfg.env_params["fixed_ee_roll"]
        self._prev_ee_yaw_cmd[env_ids] = self._ee_yaw_cmd[env_ids]
        self._prev_ee_pitch_cmd[env_ids] = self._ee_pitch_cmd[env_ids]
        self._ee_kp_cmd_plb[env_ids] = neutral_ee_kp_cmd

        # Initialize gripper command open at the beginning of the episode.
        self._gripper_cmd_norm[env_ids] = -1.0
        self._gripper_cmd_pos[env_ids] = self.cfg.env_params["gripper_open_pos"]

        self._update_task_buffers()

        # Important:
        # Fill LL history with current frame, matching successful sim2sim history layout.
        ll_frame = self._build_ll_obs_frame()
        self._ll_obs_history[env_ids] = ll_frame[env_ids].unsqueeze(1).repeat(
            1,
            self.cfg.env_params["ll_obs_history"],
            1,
        )

        # Fill HL actor/critic history with the first valid episode observation.
        actor_frame, critic_frame = self._build_hl_obs_frames()
        self._obs_history[env_ids] = actor_frame[env_ids].unsqueeze(1).repeat(
            1, self.cfg.env_params["obs_history"], 1
        )
        self._critic_history[env_ids] = critic_frame[env_ids].unsqueeze(1).repeat(
            1, self.cfg.env_params["critic_obs_history"], 1
        )

        extras = {}
        for key in self._episode_sums.keys():
            extras["Reward/" + key] = torch.mean(average_step_reward[key])
            self._episode_sums[key][env_ids] = 0.0

        for key in self._episode_metrics.keys():
            extras["Episode/" + key] = torch.mean(average_step_metric[key])
            self._episode_metrics[key][env_ids] = 0.0

        num_reset_envs = len(env_ids)
        extras["Episode_Termination/base_contact"] = (
            100.0 * torch.count_nonzero(base_contact_reset_envs).item() / max(num_reset_envs, 1)
        )
        extras["Episode_Termination/grasp_error_oob"] = (
            100.0
            * torch.count_nonzero(grasp_error_oob_reset_envs).item()
            / max(num_reset_envs, 1)
        )
        extras["Episode_Termination/s2_grasp_lost"] = (
            100.0
            * torch.count_nonzero(s2_grasp_lost_reset_envs).item()
            / max(num_reset_envs, 1)
        )
        extras["Episode_Termination/time_out"] = (
            100.0 * torch.count_nonzero(self.reset_time_outs[env_ids]).item() / max(num_reset_envs, 1)
        )
        extras["Episode/final_grasp_error_at_reset"] = torch.mean(final_grasp_error_reset_envs)
        extras["Episode/final_retrieval_target_error"] = torch.mean(
            final_retrieval_target_error_reset_envs
        )
        extras["Episode/final_object_plb_x"] = torch.mean(
            final_object_pos_plb_reset_envs[:, 0]
        )
        extras["Episode/final_object_plb_y"] = torch.mean(
            final_object_pos_plb_reset_envs[:, 1]
        )
        extras["Episode/curriculum_final_stage"] = torch.mean(
            final_curriculum_stage_reset_envs.float()
        )
        for stage_id in range(3):
            extras[f"Episode/curriculum_final_stage_s{stage_id}_percent"] = (
                100.0
                * torch.count_nonzero(
                    final_curriculum_stage_reset_envs == stage_id
                ).item()
                / max(num_reset_envs, 1)
            )
        extras["Episode/retrieval_success_percent"] = (
            100.0
            * torch.count_nonzero(episode_retrieval_success_reset_envs).item()
            / max(num_reset_envs, 1)
        )
        extras["Episode/curriculum_complete_episode_streak"] = torch.mean(
            self._curriculum_complete_episode_streak[env_ids].float()
        )
        extras["Episode/curriculum_reward_scales_frozen_percent"] = (
            100.0
            * torch.count_nonzero(
                self._curriculum_reward_scales_frozen[env_ids]
            ).item()
            / max(num_reset_envs, 1)
        )
        for reward_idx, reward_name in enumerate(self._curriculum_reward_names):
            extras[f"Episode/curriculum_final_weight_{reward_name}"] = torch.mean(
                final_curriculum_weights_reset_envs[:, reward_idx]
            )
            extras[f"Episode/curriculum_next_weight_{reward_name}"] = torch.mean(
                self._curriculum_reward_weights[env_ids, reward_idx]
            )

        self.extras["log"] = extras