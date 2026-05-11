from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

import math

from isaaclab_assets.robots.unitree import UNITREE_B2WZ1_HIGHGAINS_PITCH_CFG
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
class UnitreeB2WZ1PLBWholeBodyLocoManipObservationCfg:
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
            noise=Unoise(n_min=-0.6, n_max=0.6)
        )
        joint_vel_arm = ObsTerm(
            func=mdp.joint_vel_rel, 
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.25, n_max=0.25)
        )
        joint_vel_wheel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.6, n_max=0.6)
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5
    
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
        ee_kp_error = ObsTerm(
            func=mdp.ee_kp_error_plb,
            params={
                "command_name": "ee_kp",
                "asset_cfg": SceneEntityCfg("robot", body_names="gripperStator"),
                "kp_dx": 0.30,
                "kp_dz": 0.30,
                "ground_z": 0.0,
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
        ee_kp_phase = ObsTerm(
            func=mdp.command_phase_from_time_left,
            params={
                "command_name": "ee_kp",
                "resampling_time": 8.0,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False  # noise disabled for critic
            self.concatenate_terms = True
            self.history_length = 5
    
    # Observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class UnitreeB2WZ1PLBWholeBodyLocoManipActionCfg():
    """Action specifications for the MDP."""
    
    leg_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*"], 
        scale=0.25, 
        use_default_offset=True
    )
    
    arm_joint_pos = mdp.DelayedJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.10,
        use_default_offset=True,
        min_delay=0,
        max_delay=2,
    )

    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", 
        joint_names=[".*"], 
        scale=4.0, 
        use_default_offset=True
    )


@configclass
class UnitreeB2WZ1PLBWholeBodyLocoManipCommandsCfg():
    """Command spcifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.25,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.6, 0.6), 
            lin_vel_y=(-0.6, 0.6), 
            ang_vel_z=(-0.6, 0.6), 
            heading=(-math.pi, math.pi)
        ),
    )

    # keypoints command (kp0,kp1,kp2) in PLB, shape (N,9)
    ee_kp = mdp.PresampledKeypointsDirectCommandPLBCfg(
        asset_name="robot",
        body_name="gripperStator",
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        file_path="scripts/tools/reachable_kp0kp1kp2_plb_wholebody_v2.npy",
        sample_mode="random",
        kp_dx=0.30,
        kp_dz=0.30,
        ground_z=0.0,
    )


@configclass
class UnitreeB2WZ1PLBWholeBodyLocoManipEventCfg(EventCfg):

    # reset events
    randomize_actuator_gains_base = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_actuator_gains_arm = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    joint_armature_joint1 = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint1"]),
            "armature_distribution_params": (0.02, 0.08),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    joint_armature_lift = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint2", "joint3"]),
            "armature_distribution_params": (0.01, 0.05),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    joint_armature_wrist = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint4", "joint5", "joint6"]),
            "armature_distribution_params": (0.0, 0.02),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    joint_friction_joint1 = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint1"]),
            "friction_distribution_params": (0.01, 0.05),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    joint_friction_lift = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint2", "joint3"]),
            "friction_distribution_params": (0.005, 0.03),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    joint_friction_wrist = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint4", "joint5", "joint6"]),
            "friction_distribution_params": (0.0, 0.02),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    # Arm event
    add_ee_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripperStator"),
            "mass_distribution_params": (-0.05, 0.5),
            "operation": "add",
        },
    )

    # Randomize arm joint positions at reset to encourage exploration
    reset_arm_joint_offset_joint16 = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint1", "joint6"]),
            "position_range": (-1.0, 1.0), 
            "velocity_range": (0.0, 0.0),
        }
    )

    reset_arm_joint_offset_joint25 = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint2", "joint5"]),
            "position_range": (-0.8, 0.8), 
            "velocity_range": (0.0, 0.0),
        }
    )

    reset_arm_joint_offset_joint3 = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint3"]),
            "position_range": (-1.0, 0.4), 
            "velocity_range": (0.0, 0.0),
        }
    )

    reset_arm_joint_offset_joint4 = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint4"]),
            "position_range": (-0.4, 1.2), 
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
            "torque_range": (-0.0, 0.0),
        },
    )


@configclass
class UnitreeB2WZ1PLBWholeBodyLocoManipRewardsCfg(RewardsCfg):
    """Reward specifications for the MDP."""

    # -- arm task rewards --
    ee_kp_tracking_exp_coarse = RewTerm(
        func=manipulation_mdp.ee_kp_tracking_exp_saturated_std_schedule_plb,
        weight=1.75,
        params={
            "command_name": "ee_kp",
            "asset_cfg": SceneEntityCfg("robot", body_names="gripperStator"),
            "kp_dx": 0.30,
            "kp_dz": 0.30,
            "ground_z": 0.0,
            "w0": 1.2,
            "w1": 1.0,
            "w2": 1.0,
            "std_start": 1.0,
            "std_end": 0.50,
            "threshold": 0.15,
            "gate_start_ratio": 0.0,
            "gate_end_ratio": 0.30,
            "gate_kind": "smootherstep",
        },
    )

    ee_kp_tracking_exp_finetune_delayed = RewTerm(
        func=manipulation_mdp.ee_kp_tracking_delayed_exp_plb,
        weight=2.75,
        params={
            "command_name": "ee_kp",
            "asset_cfg": SceneEntityCfg("robot", body_names="gripperStator"),
            "kp_dx": 0.30,
            "kp_dz": 0.30,
            "ground_z": 0.0,
            "w0": 1.2,
            "w1": 1.0,
            "w2": 1.0,
            "std": 0.15,
            "std0": 0.15,
            "std1": 0.15,
            "std2": 0.15,
            "gate_start_ratio": 0.50,
            "gate_end_ratio": 0.75,
            "gate_kind": "smootherstep",
        },
    )

    ee_kp_sparse_success = RewTerm(
        func=manipulation_mdp.ee_kp_tracking_sparse_success_plb,
        weight=1.25,
        params={
            "command_name": "ee_kp",
            "asset_cfg": SceneEntityCfg("robot", body_names="gripperStator"),
            "kp_dx": 0.30,
            "kp_dz": 0.30,
            "ground_z": 0.0,
            "w0": 1.2,
            "w1": 1.0,
            "w2": 1.0,
            "th1": 0.09,
            "th2": 0.07,
            "th3": 0.05,
            "th4": 0.03,
            "bonus1": 0.2,
            "bonus2": 0.2,
            "bonus3": 0.2,
            "bonus4": 0.2,
            "metric": "weighted_mean",
        },
    )

    # -- root penalties
    body_lin_acc_l2 = RewTerm(
        func=mdp.body_lin_acc_l2,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    )
    ang_vel_x_l2 = RewTerm(
        func=mdp.ang_vel_x_l2,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_xy_vel_stand_still = RewTerm(
        func=mdp.base_xy_vel_stand_still_l2,
        weight=-5.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.06,
            "transition_width": 0.03,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # -- leg joint penalties
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.003)
    joint_power = RewTerm(
        func=mdp.joint_power,
        weight=-3.0e-04,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )
    stand_still = RewTerm(
        func=mdp.stand_still_joint_deviation_l1_smooth_weighted,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.06,
            "transition_width": 0.03,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "front_thigh_calf_scale": 0.15,
            "other_joint_scale": 1.0,
        },
    )

    joint_pos_penalty = RewTerm(
        func=mdp.joint_pos_penalty_weighted,
        weight=-0.25,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 1.0,
            "velocity_threshold": 0.2,
            "command_threshold": 0.1,
            "front_thigh_calf_scale": 0.15,
            "other_joint_scale": 1.0,
        },
    )

    # -- arm penalties
    arm_dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=-2.5e-04, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    arm_dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=-5.0e-06, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    arm_dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-10.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    arm_dof_pos_margin = RewTerm(
        func=mdp.joint_pos_soft_limit_margin,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "margin_ratio": 0.10,
            "power": 2.0,
        }
    )
    arm_dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=-0.02, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )

    # end-effector smoothness penalties
    ee_lin_vel_penalty = RewTerm(
        func=mdp.ee_lin_vel_l2,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripperStator"),
            "frame": "b",
            "axes": "xyz",
        },
    )

    ee_ang_vel_penalty = RewTerm(
        func=mdp.ee_ang_vel_l2,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripperStator"),
            "frame": "b",
            "axes": "xyz",
        },
    )

    ee_lin_acc_penalty = RewTerm(
        func=mdp.ee_lin_acc_l2,
        weight=-0.0001,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripperStator"),
            "frame": "b",
            "axes": "xyz",
        },
    )

    # -- wheel penalties
    dof_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=-2.5e-04, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    dof_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=-0.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    dof_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    
    # -- action penalties --
    action_rate_leg_l2 = RewTerm(func=mdp.action_rate_l2_leg, weight=-0.05)
    action_rate_wheel_l2 = RewTerm(func=mdp.action_rate_l2_wheel, weight=-0.05)
    action_rate_arm_l2 = RewTerm(func=mdp.action_rate_l2_arm, weight=-0.05)
    arm_action_l2 = RewTerm(func=mdp.action_l2_arm, weight=-0.0)

    # -- other penalties --
    arm_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-3.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["link00", "link01", "link02", "link03", "link04",
                                                        "link05", "link06", "gripperStator", "gripperMover"]), "threshold": 1.0},
    )


@configclass
class UnitreeB2WZ1PLBWholeBodyLocoManipTerminationCfg():
    """Termination specifications for the MDP."""
    
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 30.0},
    )


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
class UnitreeB2WZ1PLBWholeBodyLocoManipCurriculumCfg():
    """Curriculum specifications for the MDP."""

    ee_lin_vel_weight_200 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.ee_lin_vel_penalty.weight",
            "modify_fn": ramp_value,
            "modify_params": {"start": -0.05, "end": -1.25, "start_step": 4800, "end_step": 12000},
        },
    )
    
    ee_ang_vel_weight_200 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.ee_ang_vel_penalty.weight",
            "modify_fn": ramp_value,
            "modify_params": {"start": -0.01, "end": -0.25, "start_step": 4800, "end_step": 12000},
        },
    )

    ee_lin_acc_weight_200 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.ee_lin_acc_penalty.weight",
            "modify_fn": ramp_value,
            "modify_params": {"start": -0.0001, "end": -0.0075, "start_step": 4800, "end_step": 12000},
        },
    )

    arm_dof_acc_weight_200 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.arm_dof_acc_l2.weight",
            "modify_fn": ramp_value,
            "modify_params": {"start": -5.0e-06, "end": -5.0e-05, "start_step": 4800, "end_step": 12000},  # -1,0e-05
        },   
    )

    arm_dof_vel_weight_200 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.arm_dof_vel_l2.weight",
            "modify_fn": ramp_value,
            "modify_params": {"start": -0.02, "end": -0.25, "start_step": 4800, "end_step": 12000},
        },
    )

    action_rate_arm_weight_200 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.action_rate_arm_l2.weight",
            "modify_fn": ramp_value,
            "modify_params": {"start": -0.05, "end": -0.10, "start_step": 4800, "end_step": 12000},
        },
    )

    action_rate_leg_weight_200 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.action_rate_leg_l2.weight",
            "modify_fn": ramp_value,
            "modify_params": {"start": -0.05, "end": -0.075, "start_step": 4800, "end_step": 12000},
        },
    )

    action_rate_wheel_weight_200 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.action_rate_wheel_l2.weight",
            "modify_fn": ramp_value,
            "modify_params": {"start": -0.05, "end": -0.075, "start_step": 4800, "end_step": 12000},
        },
    )

    base_lin_vel_std_800 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.track_lin_vel_xy_exp.params.std",
            "modify_fn": ramp_value,
            "modify_params": {"start": 0.50, "end": 0.30, "start_step": 19200, "end_step": 36000},
        },
    )

    base_ang_vel_std_800 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.track_ang_vel_z_exp.params.std",
            "modify_fn": ramp_value,
            "modify_params": {"start": 0.50, "end": 0.25, "start_step": 19200, "end_step": 36000},
        },
    )


@configclass
class UnitreeB2WZ1PLBWholeBodyLocoManipEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for Unitree B2WZ1 loco-manipulation tracking environment."""

    # Basic settings
    observations: UnitreeB2WZ1PLBWholeBodyLocoManipObservationCfg = UnitreeB2WZ1PLBWholeBodyLocoManipObservationCfg()
    actions: UnitreeB2WZ1PLBWholeBodyLocoManipActionCfg = UnitreeB2WZ1PLBWholeBodyLocoManipActionCfg()
    commands: UnitreeB2WZ1PLBWholeBodyLocoManipCommandsCfg = UnitreeB2WZ1PLBWholeBodyLocoManipCommandsCfg()
    # MDP settings
    events: UnitreeB2WZ1PLBWholeBodyLocoManipEventCfg = UnitreeB2WZ1PLBWholeBodyLocoManipEventCfg()
    rewards: UnitreeB2WZ1PLBWholeBodyLocoManipRewardsCfg = UnitreeB2WZ1PLBWholeBodyLocoManipRewardsCfg()
    terminations: UnitreeB2WZ1PLBWholeBodyLocoManipTerminationCfg = UnitreeB2WZ1PLBWholeBodyLocoManipTerminationCfg()
    curriculum: UnitreeB2WZ1PLBWholeBodyLocoManipCurriculumCfg = UnitreeB2WZ1PLBWholeBodyLocoManipCurriculumCfg()


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
        self.scene.robot = UNITREE_B2WZ1_HIGHGAINS_PITCH_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
        self.scene.height_scanner.prim_path = '{ENV_REGEX_NS}/Robot/base_link'

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
        self.actions.arm_joint_pos.joint_names = self.arm_joint_names

        # --- command settings ---

        # --- termination settings ---

        # --- event settings ---
        self.events.physics_material.params["static_friction_range"] = (0.4, 2.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.4, 2.0)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.2)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 2.0)
        self.events.base_com.params["asset_cfg"].body_names = "base_link"
        self.events.base_com.params["com_range"] = {
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (-0.02, 0.02)
        }
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"
        self.events.base_external_force_torque.params["force_range"] = (-10.0, 10.0)
        self.events.base_external_force_torque.params["torque_range"] = (-3.0, 3.0)
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (0.0, 0.1),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-3.14, 3.14)
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
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
        self.events.reset_robot_joints.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=self.leg_joint_names)
        self.events.reset_robot_joints.params["position_range"] = (0.8, 1.2)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.randomize_actuator_gains_base.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=self.leg_joint_names+self.wheel_joint_names)
        self.events.randomize_actuator_gains_arm.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=self.arm_joint_names)

        # --- reward settings ---
        # task rewards
        self.rewards.track_lin_vel_xy_exp.weight = 4.0
        self.rewards.track_ang_vel_z_exp.weight = 2.5

        # root penalties
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.0
        self.rewards.flat_orientation_l2.weight = -0.0

        # leg joint penalties
        self.rewards.dof_torques_l2.weight = -1.0e-05
        self.rewards.dof_torques_l2.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names)}
        self.rewards.dof_vel_l2.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names)}
        self.rewards.dof_acc_l2.weight = -7.5e-06
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
        self.rewards.arm_dof_pos_margin.params["asset_cfg"].joint_names = self.arm_joint_names
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
        self.rewards.feet_air_time = None


@configclass
class UnitreeB2WZ1PLBWholeBodyLocoManipEnvCfg_PLAY(UnitreeB2WZ1PLBWholeBodyLocoManipEnvCfg):

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