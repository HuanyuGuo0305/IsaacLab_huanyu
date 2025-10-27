from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg


from isaaclab_assets.robots.unitree import UNITREE_B2_CFG  # isort: skip (Robot configuration)
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import MySceneCfg
from isaaclab.sensors.imu.imu_cfg import ImuCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise 
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.envs.common import ViewerCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

@configclass
class UnitreeB2SceneCfg(MySceneCfg):  # Parent class defines terrain, robot (None), sensors, and lighting
    # IMU sensor
    imu_sensor = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/imu_link",
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
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
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
    all_feet_over_air = DoneTerm(
        func=mdp.all_feet_over_air,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    illegal_body_slant = DoneTerm(
        func=mdp.illegal_body_slant)

        
@configclass
class UnitreeB2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    
    scene: UnitreeB2SceneCfg = UnitreeB2SceneCfg(num_envs=4096, env_spacing=2.5)
    # commands: for this task, use parent class commands
    observations: UnitreeB2ObservationCfg = UnitreeB2ObservationCfg()
    terminations: TerminationCfg = TerminationCfg()
    
    # Viewer
    viewer = ViewerCfg(origin_type="env", asset_name="Robot", body_name="base_link")
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # -----scene settings-----
        # set robot and height scanner
        self.scene.robot = UNITREE_B2_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
        self.scene.height_scanner.prim_path = '{ENV_REGEX_NS}/Robot/base_link'
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # ----- command settings -----
        self.commands.base_velocity.heading_command = False
        
        # ----- event settings -----
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.base_com.params["asset_cfg"].body_names = "base_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"
        self.events.push_robot.params = {
            "velocity_range":{
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (-1.0, 1.0),
                "roll": (-1.0, 1.0),
                "pitch": (-1.0, 1.0),
                "yaw": (-1.0, 1.0),
            }
        }
        
        # ----- reward settings -----
        # -- task
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        # -- penalties
        self.rewards.lin_vel_z_l2.weight = -4.0
        self.rewards.dof_torques_l2.weight = -5.0e-06
        self.rewards.dof_acc_l2.weight = -1.0e-06
        self.rewards.action_rate_l2.weight = -0.05
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.undesired_contacts.params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*(thigh|calf)"), "threshold":1.0}
        self.rewards.undesired_contacts.weight = -1.0
        # -- optional penalties
        self.rewards.flat_orientation_l2.weight= -5.0


@configclass
class UnitreeB2RoughEnvCfg_PLAY(UnitreeB2RoughEnvCfg):
    # Viewer
    viewer = ViewerCfg(origin_type="world", asset_name="Robot", body_name="base_link")
    
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