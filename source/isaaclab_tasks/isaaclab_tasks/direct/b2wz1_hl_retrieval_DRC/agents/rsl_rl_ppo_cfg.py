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
class RslRlMultiHeadBetaActorCriticCfg(RslRlPpoActorCriticCfg):
    """Configuration for the shared-trunk multi-head Beta actor-critic."""

    class_name: str = "MultiHeadBetaActorCritic"

    # Action partition:
    # base: 3, arm: 5, gripper: 1
    base_action_dim: int = 3
    arm_action_dim: int = 5
    gripper_action_dim: int = 1

    # Independent hidden layers after the shared actor trunk.
    base_head_hidden_dims: list[int] = [128, 128, 128]
    arm_head_hidden_dims: list[int] = [128, 128, 128]
    gripper_head_hidden_dims: list[int] = [128, 128]


@configclass
class RslRlMultiHeadPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO configuration with independent entropy coefficients per actor head."""

    # These fields are consumed by the modified rsl_rl.algorithms.PPO.
    base_entropy_coef: float = 0.0005
    arm_entropy_coef: float = 0.001
    gripper_entropy_coef: float = 0.002


@configclass
class B2WZ1HLBallPickUpPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Full 9-D multi-head Beta-PPO mobile-manipulation grasp configuration."""

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100

    experiment_name = "unitree_b2wz1_hl_beta_ball_pick_up"
    run_name = "multi_head_beta_9d_grouped_entropy"

    logger = "wandb"
    wandb_entity = "huanyu_crl"
    wandb_project = "unitree_b2wz1_hl_beta_ball_pick_up"

    # New actor architecture: do not load a standard ActorCritic checkpoint.
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    empirical_normalization = False

    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
    }

    # Beta actions are natively bounded in (-1, 1).
    clip_actions = 1.0

    policy = RslRlMultiHeadBetaActorCriticCfg(
        class_name="MultiHeadBetaActorCritic",

        # Gaussian-only compatibility fields.
        init_noise_std=1.0,
        noise_std_type="scalar",
        state_dependent_std=False,

        # Shared actor trunk:
        # actor_obs -> 512 -> 256 -> 128
        actor_hidden_dims=[512, 256, 128],

        # Critic remains a single MLP:
        # critic_obs -> 512 -> 256 -> 128 -> value
        critic_hidden_dims=[512, 256, 128],

        activation="elu",

        # Beta distribution configuration.
        distribution_type="beta",
        beta_min_concentration=1.0,
        beta_init_concentration=2.0,
        beta_eps=1.0e-6,
        beta_inference_mode="mean",

        # Action partition: 3 + 5 + 1 = 9.
        base_action_dim=3,
        arm_action_dim=5,
        gripper_action_dim=1,

        # Independent heads after shared latent.
        base_head_hidden_dims=[128, 128, 128],
        arm_head_hidden_dims=[128, 128, 128],
        gripper_head_hidden_dims=[128, 128],

        actor_obs_normalization=True,
        critic_obs_normalization=True,
    )

    algorithm = RslRlMultiHeadPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,

        # Fallback coefficient for standard policies.
        # For MultiHeadBetaActorCritic, PPO uses the three coefficients below.
        entropy_coef=0.001,

        # Independent entropy coefficients.
        #
        # Each head entropy is already averaged over that head's action
        # dimensions inside MultiHeadBetaActorCritic.
        base_entropy_coef=0.0005,
        arm_entropy_coef=0.015,
        gripper_entropy_coef=0.015,

        num_learning_epochs=5,
        num_mini_batches=4,

        learning_rate=1.0e-4,
        schedule="adaptive",
        desired_kl=0.01,

        gamma=0.995,
        lam=0.95,
        max_grad_norm=1.0,

        normalize_advantage_per_mini_batch=True,

        rnd_cfg=None,
        symmetry_cfg=None,
    )