from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg


from isaaclab_assets.robots.unitree import UNITREE_B2W_CFG
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise 
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import EventCfg, RewardsCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.envs.common import ViewerCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class UnitreeB2WObservationCfg():
    """Observation specifications for the MDP."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observation for the policy group."""
        
        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, 
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        actions = ObsTerm(func=mdp.last_action)
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5
    
    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""
        
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "base_velocity"})
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
            self.history_length = 5
    
    # Observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
        

@configclass
class UnitreeB2WActionCfg():
    """Action specifications for the MDP."""
    
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)

    joint_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=[".*"], scale=5.0, use_default_offset=True)
    

@configclass
class UnitreeB2WTerminationCfg():
    """Termination specifications for the MDP."""
    
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold":1.0},
    )
    hip_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_hip"), "threshold":1.0},
    )
    bad_orirentation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class UnitreeB2WEventCfg(EventCfg):
    
    # reset events
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "stiffness_distribution_params": (0.7, 1.3),
            "damping_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class UnitreeB2WRewardsCfg(RewardsCfg):
    """Reward specifications for the MDP."""
    
    # -- root penalties
    body_lin_acc_l2 = RewTerm(
        func=mdp.body_lin_acc_l2,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    )

    # -- joint penalties
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.00125)
    stand_still = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        },
    )
    joint_pos_penalty = RewTerm(
        func=mdp.joint_pos_penalty,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
            "command_threshold": 0.1,
        }
    )
    joint_power = RewTerm(
        func=mdp.joint_power,
        weight=-1.0e-05,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )
    joint_mirror = RewTerm(
        func=mdp.joint_mirror,
        weight=-0.025,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints":[["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"], 
                             ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],]
        },
    )
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
    
    # -- wheel penalties --
    dof_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )
    dof_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )
    dof_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=-2.5e-10, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )
    
    # -- others --
    upward = RewTerm(func=mdp.upward, weight=1.0)



@configclass
class UnitreeB2WRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    
    observations: UnitreeB2WObservationCfg = UnitreeB2WObservationCfg()
    actions: UnitreeB2WActionCfg = UnitreeB2WActionCfg()
    terminations: UnitreeB2WTerminationCfg = UnitreeB2WTerminationCfg()
    events: UnitreeB2WEventCfg = UnitreeB2WEventCfg()
    rewards: UnitreeB2WRewardsCfg = UnitreeB2WRewardsCfg()
    
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
    joint_names = leg_joint_names + wheel_joint_names

    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # --- scene settings ---
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.robot = UNITREE_B2W_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
        self.scene.height_scanner.prim_path = '{ENV_REGEX_NS}/Robot/base_link'
        
        # --- observation settings ---
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.leg_joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.leg_joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        
        # --- action settings ---
        self.actions.joint_pos.joint_names = self.leg_joint_names
        self.actions.joint_vel.joint_names = self.wheel_joint_names
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        
        # --- termination settings ---
        
        # --- event settings ---
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 1.2)
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
        self.events.reset_robot_joints.params["position_range"] = (0.8, 1.2)
        self.events.reset_robot_joints.params["velocity_range"] = (0.8, 1.2)
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
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (-1.0, 1.0),
                "roll": (-1.0, 1.0),
                "pitch": (-1.0, 1.0),
                "yaw": (-1.0, 1.0)
            }
        }
        
        # --- reward settings ---
        # task
        self.rewards.track_lin_vel_xy_exp.weight = 3.25
        self.rewards.track_ang_vel_z_exp.weight = 1.75
        
        # root penalties
        self.rewards.lin_vel_z_l2.weight = -1.75
        self.rewards.ang_vel_xy_l2.weight = -0.15
        self.rewards.flat_orientation_l2.weight = -2.0
        
        # joint penalties
        self.rewards.dof_torques_l2.weight = -1.0e-5
        self.rewards.dof_torques_l2.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names)}
        self.rewards.dof_vel_l2.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names)}
        self.rewards.dof_acc_l2.weight = -2.5e-07
        self.rewards.dof_acc_l2.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names)}
        self.rewards.dof_pos_limits.weight = -3.0
        self.rewards.dof_pos_limits.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names)}
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.leg_joint_names
        
        # wheel joint penalties
        self.rewards.dof_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.dof_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.dof_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        
        # action penalties
        self.rewards.action_rate_l2.weight = -0.05
        
        # contact penalties
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold":1.0}
        
        # others
        self.rewards.feet_air_time.params["threshold"] = 0.6
        self.rewards.feet_air_time.weight = 2.0


@configclass
class UnitreeB2WRoughEnvCfg_PLAY(UnitreeB2WRoughEnvCfg):
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.push_robot = None
        self.events.base_external_force_torque = None

        # Change terrain to flat for visualization
        USE_FLAT_TERRAIN = True
        
        if USE_FLAT_TERRAIN:
            self.scene.terrain.terrain_type = "plane"
            self.scene.terrain.terrain_generator = None
            self.curriculum.terrain_levels = None