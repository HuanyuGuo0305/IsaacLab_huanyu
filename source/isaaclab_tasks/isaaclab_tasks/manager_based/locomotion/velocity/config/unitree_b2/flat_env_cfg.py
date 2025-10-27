from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeB2RoughEnvCfg


@configclass
class UnitreeB2FlatEnvCfg(UnitreeB2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # ----- override rewards -----
        # --override rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        # --penalties
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.dof_torques_l2.weight = -5.0e-06
        self.rewards.feet_air_time.weight = 2.0
        self.rewards.dof_acc_l2.weight = -1.0e-06
        self.rewards.action_rate_l2.weight = -0.05
        self.rewards.undesired_contacts.weight = -1.0
        # --optional penalties
        self.rewards.flat_orientation_l2.weight = -5.0
        
        # ----- terrain settings -----
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        
        # ----- comand settings -----
        self.commands.base_velocity.heading_command = False

 
@configclass
class UnitreeB2FlatEnvCfg_PLAY(UnitreeB2FlatEnvCfg):
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