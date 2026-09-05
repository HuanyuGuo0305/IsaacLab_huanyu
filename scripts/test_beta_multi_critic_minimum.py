"""Minimum tests for Beta PPO + multi-critic.

Run this inside the same Python environment that imports your modified rsl_rl.

Checks:
1. Actor outputs Beta actions in (-1, 1).
2. Critic output shape is [batch, num_value_heads].
3. Multi-head GAE shapes are correct.
4. An inactive reward head is excluded from actor advantage.
5. An active reward head contributes to actor advantage.
6. Single-critic compatibility is preserved.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage


def build_obs(num_envs: int, obs_dim: int) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(num_envs, obs_dim),
            "critic": torch.randn(num_envs, obs_dim),
        },
        batch_size=[num_envs],
    )


def test_actor_critic() -> None:
    torch.manual_seed(1)

    num_envs = 64
    obs_dim = 24
    action_dim = 9
    num_heads = 2

    obs = build_obs(num_envs, obs_dim)

    policy = ActorCritic(
        obs=obs,
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=action_dim,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        distribution_type="beta",
        beta_min_concentration=1.0,
        beta_init_concentration=2.0,
        beta_eps=1.0e-6,
        beta_inference_mode="mean",
        state_dependent_std=False,
        num_value_heads=num_heads,
    )

    actions = policy.act(obs)
    values = policy.evaluate(obs)

    assert actions.shape == (num_envs, action_dim)
    assert values.shape == (num_envs, num_heads)
    assert torch.all(actions > -1.0)
    assert torch.all(actions < 1.0)
    assert torch.isfinite(actions).all()
    assert torch.isfinite(values).all()

    print("actor action shape:", tuple(actions.shape))
    print("critic value shape:", tuple(values.shape))
    print("actor/critic shape test passed")


def fill_storage(
    storage: RolloutStorage,
    reward_head_0: torch.Tensor,
    reward_head_1: torch.Tensor,
) -> None:
    num_steps, num_envs = reward_head_0.shape

    for step in range(num_steps):
        transition = RolloutStorage.Transition()
        transition.observations = TensorDict(
            {
                "policy": torch.randn(num_envs, 8),
                "critic": torch.randn(num_envs, 8),
            },
            batch_size=[num_envs],
        )
        transition.actions = torch.zeros(num_envs, 3)
        transition.rewards = (
            reward_head_0[step] + reward_head_1[step]
        ).unsqueeze(-1)
        transition.reward_groups = torch.stack(
            (reward_head_0[step], reward_head_1[step]),
            dim=-1,
        )
        transition.reward_group_active = (
            transition.reward_groups.abs() > 0.0
        )
        transition.dones = torch.zeros(num_envs, 1, dtype=torch.uint8)
        transition.values = torch.zeros(num_envs, 2)
        transition.actions_log_prob = torch.zeros(num_envs)
        transition.action_mean = torch.full((num_envs, 3), 2.0)
        transition.action_sigma = torch.full((num_envs, 3), 2.0)

        storage.add_transitions(transition)


def test_inactive_head_mask() -> None:
    torch.manual_seed(2)

    num_steps = 6
    num_envs = 32

    obs = TensorDict(
        {
            "policy": torch.zeros(num_envs, 8),
            "critic": torch.zeros(num_envs, 8),
        },
        batch_size=[num_envs],
    )

    storage = RolloutStorage(
        training_type="rl",
        num_envs=num_envs,
        num_transitions_per_env=num_steps,
        obs=obs,
        actions_shape=[3],
        device="cpu",
        num_value_heads=2,
    )

    reward_0 = torch.randn(num_steps, num_envs).abs() * 0.1
    reward_1 = torch.zeros(num_steps, num_envs)

    fill_storage(storage, reward_0, reward_1)

    storage.compute_returns(
        last_values=torch.zeros(num_envs, 2),
        gamma=0.99,
        lam=0.95,
        normalize_advantage=True,
        min_active_samples=1,
    )

    print("inactive test active heads:", storage.active_value_heads.tolist())
    print(
        "inactive test reward counts:",
        storage.reward_group_active_counts.tolist(),
    )
    print(
        "inactive test actor adv std per head:",
        storage.actor_advantage_std_per_head.tolist(),
    )

    assert storage.active_value_heads.tolist() == [True, False]
    assert storage.reward_group_active_counts[1].item() == 0
    assert storage.actor_advantage_std_per_head[1].item() == 0.0
    assert torch.isfinite(storage.advantages).all()
    assert storage.advantages.std(unbiased=False).item() > 0.9

    print("inactive-head mask test passed")


def test_active_second_head() -> None:
    torch.manual_seed(3)

    num_steps = 6
    num_envs = 32

    obs = TensorDict(
        {
            "policy": torch.zeros(num_envs, 8),
            "critic": torch.zeros(num_envs, 8),
        },
        batch_size=[num_envs],
    )

    storage = RolloutStorage(
        training_type="rl",
        num_envs=num_envs,
        num_transitions_per_env=num_steps,
        obs=obs,
        actions_shape=[3],
        device="cpu",
        num_value_heads=2,
    )

    reward_0 = torch.randn(num_steps, num_envs).abs() * 0.1
    reward_1 = torch.zeros(num_steps, num_envs)
    reward_1[-1, :8] = 1.0

    fill_storage(storage, reward_0, reward_1)

    storage.compute_returns(
        last_values=torch.zeros(num_envs, 2),
        gamma=0.99,
        lam=0.95,
        normalize_advantage=True,
        min_active_samples=4,
    )

    print("active test active heads:", storage.active_value_heads.tolist())
    print(
        "active test reward counts:",
        storage.reward_group_active_counts.tolist(),
    )
    print(
        "active test actor adv std per head:",
        storage.actor_advantage_std_per_head.tolist(),
    )

    assert storage.active_value_heads.tolist() == [True, True]
    assert storage.reward_group_active_counts[1].item() == 8
    assert storage.actor_advantage_std_per_head[0].item() > 0.9
    assert storage.actor_advantage_std_per_head[1].item() > 0.9
    assert torch.isfinite(storage.advantages).all()

    print("active second-head test passed")


def test_single_critic_compatibility() -> None:
    num_steps = 4
    num_envs = 16

    obs = TensorDict(
        {
            "policy": torch.zeros(num_envs, 8),
            "critic": torch.zeros(num_envs, 8),
        },
        batch_size=[num_envs],
    )

    storage = RolloutStorage(
        training_type="rl",
        num_envs=num_envs,
        num_transitions_per_env=num_steps,
        obs=obs,
        actions_shape=[3],
        device="cpu",
        num_value_heads=1,
    )

    for _ in range(num_steps):
        transition = RolloutStorage.Transition()
        transition.observations = obs.clone()
        transition.actions = torch.zeros(num_envs, 3)
        transition.rewards = torch.zeros(num_envs, 1)
        transition.reward_groups = torch.zeros(num_envs, 1)
        transition.reward_group_active = torch.zeros(
            num_envs,
            1,
            dtype=torch.bool,
        )
        transition.dones = torch.zeros(num_envs, 1, dtype=torch.uint8)
        transition.values = torch.zeros(num_envs, 1)
        transition.actions_log_prob = torch.zeros(num_envs)
        transition.action_mean = torch.zeros(num_envs, 3)
        transition.action_sigma = torch.ones(num_envs, 3)
        storage.add_transitions(transition)

    storage.compute_returns(
        last_values=torch.zeros(num_envs, 1),
        gamma=0.99,
        lam=0.95,
        normalize_advantage=True,
        min_active_samples=10,
    )

    assert storage.active_value_heads.tolist() == [True]
    assert storage.advantages.shape == (num_steps, num_envs, 1)
    assert torch.isfinite(storage.advantages).all()

    print("single-critic compatibility test passed")


def main() -> None:
    test_actor_critic()
    test_inactive_head_mask()
    test_active_second_head()
    test_single_critic_compatibility()
    print("All Beta multi-critic minimum tests passed.")


if __name__ == "__main__":
    main()
