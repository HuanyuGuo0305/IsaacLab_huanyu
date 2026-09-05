# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Unitree-B2WZ1-HL-BallPickUp-S1-Direct-v0",
    entry_point=f"{__name__}.b2wz1_hl_env:B2WZ1HLBallPickUpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2wz1_hl_env:B2WZ1HLBallPickUpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1HLBallPickUpPPORunnerCfg",
    },
)