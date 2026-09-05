# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unsupervised Actuator Net (UAN) training for the Unitree Z1.

Closes the sim-to-real gap for the Z1's actuators by learning a residual
torque from real hardware data, following arXiv:2502.10894 Section II-A.

Before first use, build the Z1-only USD (the repo ships only the URDF):

    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/UAN/convert_z1_urdf.py

Then train:

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Unitree-Z1-UAN-Direct-v0 --headless
"""

import gymnasium as gym

from . import agents


def _register_uan_actor_critic() -> None:
    """Make ``UANSharedActorCritic`` findable by name from the PPO config.

    rsl_rl resolves ``policy.class_name`` with ``eval`` against the namespace
    of its runner module, which does ``from rsl_rl.modules import *`` at
    import time. A late attribute on ``rsl_rl.modules`` alone is therefore not
    enough -- that star-import already snapshotted its names -- so the class
    is injected into the runner module's globals as well.

    Import failures are swallowed on purpose: this package is imported by
    task-listing tools that never load rsl_rl, and gym registration must not
    depend on the RL library being present.
    """
    try:
        import rsl_rl.modules as rsl_rl_modules

        from .modules.uan_actor_critic import UANSharedActorCritic
    except Exception:
        return

    rsl_rl_modules.UANSharedActorCritic = UANSharedActorCritic
    if not hasattr(rsl_rl_modules, "__all__"):
        pass
    elif "UANSharedActorCritic" not in rsl_rl_modules.__all__:
        rsl_rl_modules.__all__.append("UANSharedActorCritic")

    try:
        import rsl_rl.runners.on_policy_runner as runner_module

        runner_module.UANSharedActorCritic = UANSharedActorCritic
    except Exception:
        pass


_register_uan_actor_critic()


gym.register(
    id="Isaac-Unitree-Z1-UAN-Direct-v0",
    entry_point=f"{__name__}.uan_env:UANEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.uan_env:UANEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UANPPORunnerCfg",
    },
)
