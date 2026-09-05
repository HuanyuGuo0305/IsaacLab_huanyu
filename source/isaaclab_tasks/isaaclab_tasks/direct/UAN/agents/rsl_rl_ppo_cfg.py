# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class RslRlUANActorCriticCfg(RslRlPpoActorCriticCfg):
    """Actor-critic whose actor is one network shared across all six joints."""

    class_name: str = "UANSharedActorCritic"

    # Must match UANEnvCfg: 6 joints x (2 * history_length) observations.
    num_joints: int = 6
    obs_per_joint: int = 50

    # The paper's UAN: 2-layer MLP, [128, 128], ELU, run at every sim step.
    uan_hidden_dims: list[int] = [128, 128]


@configclass
class UANPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO for Z1 unsupervised actuator net training."""

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100

    experiment_name = "unitree_z1_uan"
    run_name = "shared_per_joint_residual_torque"

    logger = "tensorboard"

    resume = False
    empirical_normalization = False

    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
    }

    policy = RslRlUANActorCriticCfg(
        class_name="UANSharedActorCritic",
        init_noise_std=0.5,
        noise_std_type="scalar",
        # Inherited fields the parent still builds against. actor_hidden_dims
        # is unused -- the actor is replaced by the shared per-joint network
        # sized by uan_hidden_dims -- but it must stay valid for construction.
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        num_joints=6,
        obs_per_joint=50,
        uan_hidden_dims=[128, 128],
        actor_obs_normalization=True,
        critic_obs_normalization=True,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        # The paper splits the actor's data into four mini-batches while
        # giving the critic the whole batch, "as we found that larger batch
        # sizes produce more stable gradients and result in lower value
        # function loss". Stock rsl_rl uses one setting for both, so this is
        # the actor's value; raise it if the value loss is noisy.
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=True,
        rnd_cfg=None,
        symmetry_cfg=None,
    )
