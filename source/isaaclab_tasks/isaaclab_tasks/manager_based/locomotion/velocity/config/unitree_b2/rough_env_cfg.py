from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg


from isaaclab_assets.robots.unitree import UNITREE_B2_CFG  # isort: skip (Robot configuration)
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import MySceneCfg, EventCfg, RewardsCfg
from isaaclab.sensors.imu.imu_cfg import ImuCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise 
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import RewardTermCfg as RewTerm

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

@configclass
class UnitreeB2SceneCfg(MySceneCfg):  # Parent class defines terrain, robot (None), sensors, and lighting
    # Height scanner base
    height_scanner_base = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.1, 0.1]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # IMU sensor
    imu_sensor = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=ImuCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        history_length=10,
        debug_vis=False,
    ) 


@configclass
class UnitreeB2ObservationCfg():
    """Observation specifications for the MDP."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group."""
        
        # observation terms (oder preserved)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        # imu = ObsTerm(func=mdp.imu_states, noise=Unoise(n_min=-0.1, n_max=0.1))
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""
        
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )
        # imu = ObsTerm(func=mdp.imu_states)
        
        def __post_init__(self):
            self.enable_corruption = False  # noise disabled for critic
            self.concatenate_terms = True
    
    # Observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class TerminationCfg():
    """Termination terms for the MDP."""
    
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold":1.0},
    )
    hip_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_hip"), "threshold":1.0},
    )
    bad_orirentation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.9})


@configclass
class UnitreeB2EventsCfg(EventCfg):
    """Event terms for the MDP."""

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


@configclass
class UnitreeB2RoughRewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    # -- root penalties
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "target_height": 0.53,   
        },
    )
    body_lin_acc_l2 = RewTerm(
        func=mdp.body_lin_acc_l2,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    )

    # A function to penalize joint position deviations
    def create_dof_deviation_l1_term(self, attr_name, weight, joint_names_pattern):
        rew_term = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=weight,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=joint_names_pattern)},
        )
        setattr(self, attr_name, rew_term)  # setattr(object, name, value) <-> object.name = value

    # -- joint penalties
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.00025)
    dof_vel_limits = RewTerm(
        func=mdp.joint_vel_limits, 
        weight=0.0, 
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "soft_ratio": 1.0},
    )
    joint_power = RewTerm(
        func=mdp.joint_power, 
        weight=0.0, 
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*")},
    )
    stand_still = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-0.1,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        },
    )
    joint_pos_penalty = RewTerm(
        func=mdp.joint_pos_penalty,
        weight=-0.2,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
            "command_threshold": 0.1,
        }
    )
    joint_mirror = RewTerm(
        func=mdp.joint_mirror,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints":[["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"], 
                             ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],]
        },
    )

    # -- contact sensor
    contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot",), "threshold": 100.0},
    )

    # -- other penalties
    feet_height_body = RewTerm(
        func=mdp.feet_height_body,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "tanh_mult": 2.0,
            "target_height": -0.4,
            "command_name": "base_velocity",
        },
    )
    upward = RewTerm(func=mdp.upward, weight=0.0)




@configclass
class UnitreeB2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    
    scene: UnitreeB2SceneCfg = UnitreeB2SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: UnitreeB2ObservationCfg = UnitreeB2ObservationCfg()
    terminations: TerminationCfg = TerminationCfg()
    events: UnitreeB2EventsCfg = UnitreeB2EventsCfg()
    rewards: UnitreeB2RoughRewardsCfg = UnitreeB2RoughRewardsCfg()
    
    # Viewer
    viewer = ViewerCfg(eye=(5.0, 5.0, 5.0), resolution=(1280, 720), origin_type="env", env_index=0, asset_name="robot")
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # -----scene settings-----
        # set robot and height scanner
        self.scene.robot = UNITREE_B2_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
        self.scene.height_scanner.prim_path = '{ENV_REGEX_NS}/Robot/base_link'
        self.scene.height_scanner_base.prim_path = '{ENV_REGEX_NS}/Robot/base_link'
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # -----action settings-----
        # reduce action scale
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}

        # ----- command settings -----
        self.commands.base_velocity.heading_command = False
        
        # ----- event settings -----
        self.events.physics_material.params["static_friction_range"] = (0.7, 0.9)
        self.events.physics_material.params["dynamic_friction_range"] = (0.5, 0.7)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.base_com.params["asset_cfg"].body_names = "base_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"
        self.events.base_external_force_torque.params["force_range"] = (-10.0, 10.0)
        self.events.base_external_force_torque.params["torque_range"] = (-5.0, 5.0)
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

        # ----- reward settings -----
        # -- task
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        # -- root penalties
        self.rewards.lin_vel_z_l2.weight = -2.75
        self.rewards.ang_vel_xy_l2.weight = -0.1   
        self.rewards.flat_orientation_l2.weight= -0.0
        # -- joint penalties
        self.rewards.dof_torques_l2.weight = -5e-06
        self.rewards.dof_acc_l2.weight = -1.5e-07
        self.rewards.dof_pos_limits.weight = -3.0
        # -- action penalties
        self.rewards.action_rate_l2.weight = -0.005
        # -- contact sensor
        self.rewards.undesired_contacts.params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold":1.0}
        self.rewards.undesired_contacts.weight = -3.0
        # -- others
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*foot"
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.weight = 3.0

        # ----- termination settings -----
        # enable the flowing termination terms in flat terrain
        self.terminations.base_contact = None
        self.terminations.hip_contact = None


@configclass
class UnitreeB2RoughEnvCfg_PLAY(UnitreeB2RoughEnvCfg):

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