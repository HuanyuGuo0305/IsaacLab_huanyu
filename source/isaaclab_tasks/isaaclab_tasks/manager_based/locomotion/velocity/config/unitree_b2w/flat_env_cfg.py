from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

from .rough_env_cfg import UnitreeB2WRoughEnvCfg


@configclass
class UnitreeB2WFlatEnvCfg(UnitreeB2WRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # no height scan
        self.scene.height_scanner = None
        self.observations.critic.height_scan = None
        
        # -- override rewards
        # task
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        # root penalties
        self.rewards.lin_vel_z_l2.weight = -1.75
        self.rewards.ang_vel_xy_l2.weight = -0.15 
        self.rewards.flat_orientation_l2.weight= -3.0 
        # joint penalties
        self.rewards.dof_torques_l2.weight = -1e-05
        self.rewards.dof_acc_l2.weight = -2.5e-07
        self.rewards.dof_pos_limits.weight = -3.0
        # action penalties
        self.rewards.action_rate_l2.weight = -0.035
        # contact sensor
        self.rewards.undesired_contacts.params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold":1.0}
        self.rewards.undesired_contacts.weight = -1.0
        # others
        self.rewards.feet_air_time.weight = 0.0

        # ----- terrain settings -----
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        
        # ----- comand settings -----
        self.commands.base_velocity.heading_command = False

        # ----- termination settings -----
        self.terminations.bad_orirentation = None

 
@configclass
class UnitreeB2WFlatEnvCfg_PLAY(UnitreeB2WFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()   
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None 