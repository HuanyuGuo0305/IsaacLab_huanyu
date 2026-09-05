# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared per-joint actor for the Unsupervised Actuator Net.

The paper trains ONE network for all six arm actuators, each processed
independently::

    "Assuming each arm joint is identical, a single UAN is shared across all
     of the arm's actuators, with each actuator being processed independently
     by the shared network. We constrain the observation space to include a
     history of the past 20 position and velocity errors for each relevant
     actuator. These design choices help prevent overfitting to other aspects
     of the training data, such as inertial coupling. Also, sharing the data
     across actuators improves data efficiency."

Two things follow, and both matter:

  - Six times the gradient signal per step, from the same rollouts.
  - The network physically cannot key on joint identity or on what the other
    joints are doing, so it cannot paper over the sim-to-real gap by learning
    this arm's particular inertial coupling.

Only the ACTOR is shared per joint. The critic stays a plain MLP over the
full flattened state, because the value of a state is a single number about
the whole arm, not something each joint owns.

Implementation note: this subclasses rsl_rl's ``ActorCritic`` and swaps in a
different ``self.actor`` module rather than reimplementing the interface.
The distribution, log-prob, entropy and checkpoint plumbing differ between
rsl_rl versions and forks, so inheriting them is the only way this stays
correct against the fork actually installed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from rsl_rl.modules import ActorCritic
    from rsl_rl.utils import resolve_nn_activation
except ImportError as exc:  # pragma: no cover - only meaningful inside Isaac
    raise ImportError(
        "UANSharedActorCritic needs rsl-rl-lib. It resolves at import time "
        "inside the Isaac Sim python environment."
    ) from exc


class SharedPerJointActor(nn.Module):
    """Maps each joint's own error history to that joint's residual torque.

    Input  ``(batch, num_joints * obs_per_joint)`` laid out joint-major.
    Output ``(batch, num_joints)``.

    The environment writes its observation as
    ``err_history.reshape(num_envs, -1)`` from a
    ``(num_envs, joints, history, 2)`` buffer, which is exactly joint-major,
    so the view below lines up with it. Get this wrong and the network is fed
    a coherent but meaningless permutation, which trains without complaint --
    hence the explicit size check in ``__init__``.
    """

    def __init__(
        self,
        num_joints: int,
        obs_per_joint: int,
        hidden_dims: list[int],
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.num_joints = int(num_joints)
        self.obs_per_joint = int(obs_per_joint)

        act = resolve_nn_activation(activation)
        layers: list[nn.Module] = []
        prev = self.obs_per_joint
        for dim in hidden_dims:
            layers += [nn.Linear(prev, dim), act]
            prev = dim
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        expected = self.num_joints * self.obs_per_joint
        if obs.shape[-1] != expected:
            raise ValueError(
                f"UAN actor expects {expected} observations "
                f"({self.num_joints} joints x {self.obs_per_joint}), got "
                f"{obs.shape[-1]}. observation_space and history_length in "
                "UANEnvCfg must match num_joints/obs_per_joint here."
            )
        batch = obs.shape[:-1]
        per_joint = obs.reshape(-1, self.num_joints, self.obs_per_joint)
        # Fold joints into the batch so one set of weights sees them all.
        flat = per_joint.reshape(-1, self.obs_per_joint)
        out = self.net(flat).reshape(-1, self.num_joints)
        return out.reshape(*batch, self.num_joints)


class UANSharedActorCritic(ActorCritic):
    """rsl_rl actor-critic whose actor is shared across joints."""

    def __init__(
        self,
        *args,
        num_joints: int = 6,
        obs_per_joint: int = 50,
        uan_hidden_dims: list[int] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        hidden = uan_hidden_dims or [128, 128]
        activation = kwargs.get("activation", "elu")
        self.num_joints = int(num_joints)
        self.obs_per_joint = int(obs_per_joint)

        # Replace the inherited MLP actor. Everything else -- distribution,
        # log-prob, entropy, normalization, checkpointing -- is inherited.
        self.actor = SharedPerJointActor(
            num_joints=self.num_joints,
            obs_per_joint=self.obs_per_joint,
            hidden_dims=hidden,
            activation=activation,
        )

        n_shared = sum(p.numel() for p in self.actor.parameters())
        print(
            f"[UAN] shared per-joint actor: {self.obs_per_joint} -> "
            f"{hidden} -> 1, applied to {self.num_joints} joints "
            f"({n_shared:,} parameters, vs a "
            f"{self.num_joints * self.obs_per_joint}-input MLP)"
        )
