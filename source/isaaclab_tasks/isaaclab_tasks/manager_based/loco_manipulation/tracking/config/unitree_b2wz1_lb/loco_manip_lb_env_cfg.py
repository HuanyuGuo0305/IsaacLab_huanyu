from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

import math

from isaaclab_assets.robots.unitree import UNITREE_B2WZ1_CFG
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import EventCfg, RewardsCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.envs.common import ViewerCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.manipulation.reach.mdp as manipulation_mdp


@configclass
class UnitreeB2WZ1LBLocoManipObservationCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observation for the policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, 
            noise=Unoise(n_min=-0.15, n_max=0.15)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "base_velocity"})
        ee_kp_commands = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "ee_kp"}
        )
        ee_current_kp = ObsTerm(
            func=mdp.ee_current_kp,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="gripperStator"),
                "kp_dx": 0.30,
                "kp_dz": 0.30,
                "frame": "lb",
            },
        )
        joint_pos_leg = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_pos_arm = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel_leg = ObsTerm(
            func=mdp.joint_vel_rel, 
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.8, n_max=0.8)
        )
        joint_vel_arm = ObsTerm(
            func=mdp.joint_vel_rel, 
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.25, n_max=0.25)
        )
        joint_vel_wheel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.8, n_max=0.8)
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 3
    
    @configclass
    class CriticCfg(ObsGroup):
        """Observation for the critic group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "base_velocity"}
        )
        ee_kp_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "ee_kp"}
        )
        ee_current_kp = ObsTerm(
            func=mdp.ee_current_kp,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="gripperStator"),
                "kp_dx": 0.30,
                "kp_dz": 0.30,
                "frame": "lb",
            },
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False  # noise disabled for critic
            self.concatenate_terms = True
            self.history_length = 3
    
    # Observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class UnitreeB2WZ1LBLocoManipActionCfg():
    """Action specifications for the MDP."""
    
    leg_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*"], 
        scale=0.25, 
        use_default_offset=True
    )
    
    arm_joint_pos_1 = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint1"],
        scale=0.25,
        use_default_offset=True,
    )

    arm_joint_pos_2 = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint2"],
        scale=0.40,
        use_default_offset=True,
    )

    arm_joint_pos_345 = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint3", "joint4", "joint5"],
        scale=0.25,
        use_default_offset=True,
    )

    arm_joint_pos_6 = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint6"],
        scale=0.20,
        use_default_offset=True,
    )

    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", 
        joint_names=[".*"], 
        scale=4.0, 
        use_default_offset=True
    )


@configclass
class UnitreeB2WZ1LBLocoManipCommandsCfg():
    """Command spcifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.05,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.8, 0.8), 
            lin_vel_y=(-0.8, 0.8), 
            ang_vel_z=(-0.6, 0.6), 
            heading=(-math.pi, math.pi)
        ),
    )

    # keypoints command (kp0,kp1,kp2) in LB, shape (N,9)
    ee_kp = mdp.PresampledKeypointsInterpolateCommandLBCfg(
        asset_name="robot",
        body_name="gripperStator",
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        file_path="scripts/tools/reachable_kp0kp1kp2_lb.npy",
        sample_mode="random",
        kp_dx=0.30,  # must match sampling cfg
        kp_dz=0.30,
        kp0_threshold=0.20,
        rot_threshold=0.40,
    )


@configclass
class UnitreeB2WZ1LBLocoManipEventCfg(EventCfg):

    # reset events
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    
    # Arm event
    add_ee_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripperStator"),
            "mass_distribution_params": (-0.1, 0.5),
            "operation": "add",
        },
    )

    # Randomize arm joint positions at reset to encourage exploration
    reset_arm_joint_offset_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "position_range": (-0.20, 0.20), 
            "velocity_range": (0.0, 0.0),
        }
    )

    # Add an external force to simulate a payload being carried
    arm_force = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripperStator"),
            "force_range": (-2.0, 2.0),
            "torque_range": (-1.0, 1.0),
        },
    )


@configclass
class UnitreeB2WZ1LBLocoManipRewardsCfg(RewardsCfg):
    """Reward specifications for the MDP."""

    # -- task (arm)
    ee_kp_tracking_delayed = RewTerm(
        func=manipulation_mdp.keypoints_command_error_exp_lb_delayed,
        weight=12.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripperStator"),
            "command_name": "ee_kp",
            "std": 0.25,
            "std0": 0.25,
            "std1": 0.25,
            "std2": 0.30,
            "kp_dx": 0.30,
            "kp_dz": 0.30,
            "w0": 1.2,
            "w1": 1.0,
            "w2": 1.0,
            "track_window_s": 5.0,  
        },
    )

    ee_kp_progress = RewTerm(
        func=manipulation_mdp.keypoints_command_progress_lb_robust_antijitter,
        weight=20.0,
        params={
            "command_name": "ee_kp",
            "asset_cfg": SceneEntityCfg("robot", body_names="gripperStator"),
            "kp_dx": 0.30,
            "kp_dz": 0.30,

            # progress only in early stage
            "active_window_s": 3.0,

            # anti-jitter
            "progress_deadband": 0.06,
            "improve_deadband": 0.002,

            # cap per-step shaping magnitude
            "pos_clip": 0.10,
            "neg_clip": 0.05,
            "w_overshoot": 0.8,
            "eps": 1e-6,
        },
    )
    
    # -- root penalties
    body_lin_acc_l2 = RewTerm(
        func=mdp.body_lin_acc_l2,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    )

    # -- leg joint penalties
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.00125)
    joint_power = RewTerm(
        func=mdp.joint_power,
        weight=-1.0e-05,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )
    stand_still = RewTerm(
        func=mdp.stand_still_joint_deviation_l1_smooth,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.05,
            "transition_width": 0.03,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )
    joint_pos_penalty = RewTerm(
        func=mdp.joint_pos_penalty,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 3.0,
            "velocity_threshold": 0.25,
            "command_threshold": 0.10,
        },
    )
    joint_mirror = RewTerm(
        func=mdp.joint_mirror,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints":[["FR_(hip|thigh|calf).*", "FL_(hip|thigh|calf).*"],
                             ["RR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],]
        },
    )

    # -- arm penalties
    arm_dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=-5.0e-5, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    arm_dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=-1.0e-6, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    arm_dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    arm_dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=-0.01, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    ee_jitter_vel = RewTerm(
        func=mdp.ee_jitter_lin_vel_l2_delayed,
        weight=-0.25,  
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripperStator"),
            "command_name": "ee_kp",
            "track_window_s": 5.0,
            "frame": "b",
            "axes": "xyz",
        },
    )
    ee_jitter_dv = RewTerm(
        func=mdp.ee_jitter_lin_vel_change_l2_delayed,
        weight=-0.20,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripperStator"),
            "command_name": "ee_kp",
            "track_window_s": 5.0,
            "frame": "b",
            "axes": "xyz",
            "clip_max": 2.0,
        },
    )

    # -- wheel penalties
    dof_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=-5.0e-05, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    dof_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=-0.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    dof_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=-2.0e-8, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    
    # -- action penalties --
    action_rate_leg_l2 = RewTerm(func=mdp.action_rate_l2_leg, weight=-0.035)
    action_rate_wheel_l2 = RewTerm(func=mdp.action_rate_l2_wheel, weight=-0.025)
    action_rate_arm_l2 = RewTerm(func=mdp.action_rate_l2_arm, weight=-0.035)
    arm_action_l2 = RewTerm(func=mdp.action_l2_arm, weight=-0.0075)
    leg_action_l2 = RewTerm(func=mdp.action_l2_leg, weight=-0.0)
    wheel_action_l2 = RewTerm(func=mdp.action_l2_wheel, weight=-0.0)

    # -- other penalties --
    arm_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["link00", "link01", "link02", "link03", "link04",
                                                        "link05", "link06", "gripperStator", "gripperMover"]), "threshold": 3.0},
    )


@configclass
class UnitreeB2WZ1LBLocoManipTerminationCfg():
    """Termination specifications for the MDP."""
    
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 30.0},
    )
    # hip_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_hip"]), "threshold": 30.0},
    # )


# curriculum helper function
def override_value(env, env_ids, data, value, num_steps):
    if env.common_step_counter > num_steps:
        return value
    return mdp.modify_term_cfg.NO_CHANGE

def ramp_value(env, env_ids, data, start, end, start_step, end_step):
    step = env.common_step_counter
    if step <= start_step:
        return start
    if step >= end_step:
        return end
    alpha = (step - start_step) / max(1, (end_step - start_step))
    return start + alpha * (end - start)


@configclass
class UnitreeB2WZ1LBLocoManipCurriculumCfg():
    """Curriculum specifications for the MDP."""

    ee_kp_tracking_weight_600 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.ee_kp_tracking_delayed.weight",
            "modify_fn": ramp_value,
            "modify_params": {"start": 12.0, "end": 20.0, "start_step": 14400, "end_step": 48000},
        },
    )

    ee_kp_track_std0_600 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.ee_kp_tracking_delayed.params.std0",
            "modify_fn": ramp_value,
            "modify_params": {"start": 0.25, "end": 0.07, "start_step": 14400, "end_step": 48000},
        },
    )

    ee_kp_track_std1_600 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.ee_kp_tracking_delayed.params.std1",
            "modify_fn": ramp_value,
            "modify_params": {"start": 0.25, "end": 0.07, "start_step": 14400, "end_step": 48000},
        },
    )

    ee_kp_track_std2_600 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.ee_kp_tracking_delayed.params.std2",
            "modify_fn": ramp_value,
            "modify_params": {"start": 0.30, "end": 0.07, "start_step": 14400, "end_step": 48000},
        },
    )

    base_vel_xy_600 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.track_lin_vel_xy_exp.params.std",
            "modify_fn": ramp_value,
            "modify_params": {"start": 0.50, "end": 0.25, "start_step": 14400, "end_step": 48000},
        },
    )

    base_vel_yaw_600 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.track_ang_vel_z_exp.params.std",
            "modify_fn": ramp_value,
            "modify_params": {"start": 0.50, "end": 0.25, "start_step": 14400, "end_step": 48000},
        },
    )

    ee_progress_w_1000 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.ee_kp_progress.weight",
            "modify_fn": ramp_value,
            "modify_params": {"start": 20.0, "end": 0.0, "start_step": 24000, "end_step": 48000},
        },
    )


@configclass
class UnitreeB2WZ1LBLocoManipEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for Unitree B2WZ1 loco-manipulation tracking environment."""

    # Basic settings
    observations: UnitreeB2WZ1LBLocoManipObservationCfg = UnitreeB2WZ1LBLocoManipObservationCfg()
    actions: UnitreeB2WZ1LBLocoManipActionCfg = UnitreeB2WZ1LBLocoManipActionCfg()
    commands: UnitreeB2WZ1LBLocoManipCommandsCfg = UnitreeB2WZ1LBLocoManipCommandsCfg()
    # MDP settings
    events: UnitreeB2WZ1LBLocoManipEventCfg = UnitreeB2WZ1LBLocoManipEventCfg()
    rewards: UnitreeB2WZ1LBLocoManipRewardsCfg = UnitreeB2WZ1LBLocoManipRewardsCfg()
    terminations: UnitreeB2WZ1LBLocoManipTerminationCfg = UnitreeB2WZ1LBLocoManipTerminationCfg()
    curriculum: UnitreeB2WZ1LBLocoManipCurriculumCfg = UnitreeB2WZ1LBLocoManipCurriculumCfg()


    # viewer
    viewer = ViewerCfg(eye=(8.0, 0.0, 10.0), lookat=(5.0, -50.0, -10.0), resolution=(1280, 720), origin_type="world")

    # joint names   
    leg_joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    wheel_joint_names = [
        "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint",
    ]
    arm_joint_names = [
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
    ]
    gripper_joint_name = [
        "jointGripper",
    ]
    

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # --- scene settings ---
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.terrain_type = "plane"
        self.scene.robot = UNITREE_B2WZ1_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
        self.scene.height_scanner.prim_path = '{ENV_REGEX_NS}/Robot/base_link'

        # no height scan
        self.scene.height_scanner = None
        self.observations.critic.height_scan = None

        # --- curriculum settings ---

        # --- observation settings ---
        self.observations.policy.joint_pos_leg.params["asset_cfg"].joint_names = self.leg_joint_names
        self.observations.policy.joint_pos_arm.params["asset_cfg"].joint_names = self.arm_joint_names
        self.observations.policy.joint_vel_leg.params["asset_cfg"].joint_names = self.leg_joint_names
        self.observations.policy.joint_vel_arm.params["asset_cfg"].joint_names = self.arm_joint_names
        self.observations.policy.joint_vel_wheel.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.leg_joint_names + self.arm_joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.leg_joint_names + self.arm_joint_names + self.wheel_joint_names

        # --- action settings ---
        self.actions.leg_joint_pos.joint_names = self.leg_joint_names
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        # --- command settings ---
        self.commands.base_velocity.heading_command = False

        # --- termination settings ---

        # --- event settings ---
        self.events.physics_material.params["static_friction_range"] = (0.6, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 1.2)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.2)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.add_base_mass.params["mass_distribution_params"] = (-5.0, 5.0)
        self.events.base_com.params["asset_cfg"].body_names = "base_link"
        self.events.base_com.params["com_range"] = {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.05, 0.05)
        }
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"
        self.events.base_external_force_torque.params["force_range"] = (-30.0, 30.0)
        self.events.base_external_force_torque.params["torque_range"] = (-10.0, 10.0)
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-3.14, 3.14)
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        self.events.push_robot.params = {
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5)
            }
        }
        self.events.reset_robot_joints.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names),
        }
        self.events.reset_robot_joints.params["position_range"] = (0.8, 1.2)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.reset_arm_joint_offset_joint.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=self.arm_joint_names)

        # --- reward settings ---
        # task rewards
        self.rewards.track_lin_vel_xy_exp.weight = 4.5
        self.rewards.track_ang_vel_z_exp.weight = 2.0

        # root penalties
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.10
        self.rewards.flat_orientation_l2.weight = -2.0

        # leg joint penalties
        self.rewards.dof_torques_l2.weight = -5.0e-06
        self.rewards.dof_torques_l2.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names)}
        self.rewards.dof_vel_l2.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names)}
        self.rewards.dof_acc_l2.weight = -2.5e-07
        self.rewards.dof_acc_l2.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names)}
        self.rewards.dof_pos_limits.weight = -3.0
        self.rewards.dof_pos_limits.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names)}
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names + self.arm_joint_names
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.leg_joint_names

        # arm joint penalties
        self.rewards.arm_dof_torques_l2.params["asset_cfg"].joint_names = self.arm_joint_names
        self.rewards.arm_dof_acc_l2.params["asset_cfg"].joint_names = self.arm_joint_names
        self.rewards.arm_dof_pos_limits.params["asset_cfg"].joint_names = self.arm_joint_names
        self.rewards.arm_dof_vel_l2.params["asset_cfg"].joint_names = self.arm_joint_names

        # wheel joint penalties
        self.rewards.dof_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.dof_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.dof_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names

        # action penalties
        self.rewards.action_rate_l2 = None

        # contact penalties
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_hip"]), "threshold":1.0}
        
        # others
        self.rewards.feet_air_time.params["threshold"] = 0.6
        self.rewards.feet_air_time.weight = 2.0


@configclass
class UnitreeB2WZ1LBLocoManipEnvCfg_PLAY(UnitreeB2WZ1LBLocoManipEnvCfg):

    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        """
        Adjustments for debugging
        """
        # disable observation corruption for play
        self.observations.policy.enable_corruption = False

        # special commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # remove events for play
        self.events = None
        
        # disable curriculum for play
        self.curriculum = None