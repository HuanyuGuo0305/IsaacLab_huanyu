import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
gym.register(
    id="Isaac-Tracking-LocoManip-UnitreeB2WZ1-PLB-Traj-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manip_plb_env_cfg:UnitreeB2WZ1PLBTrajLocoManipEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WZ1PLBTrajLocoManipPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Tracking-LocoManip-UnitreeB2WZ1-PLB-Traj-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manip_plb_env_cfg:UnitreeB2WZ1PLBTrajLocoManipEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WZ1PLBTrajLocoManipPPORunnerCfg",
    },
)