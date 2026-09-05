# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import torch


class LLPolicyWrapper:
    """Wrapper for a frozen low-level RSL-RL exported policy.

    This wrapper is designed for IsaacLab RSL-RL exported policies, for example:

        logs/rsl_rl/<experiment_name>/<load_run>/exported/policy.pt

    The exported policy.pt is expected to be a TorchScript module. In IsaacLab,
    this exported policy usually already includes the observation normalizer if
    actor_obs_normalization was enabled during training.

    Therefore, this wrapper does NOT create another normalizer.
    """

    def __init__(
        self,
        task: str,
        experiment_name: str,
        load_run: str,
        checkpoint: str,
        device: str,
    ):
        self.task = task
        self.experiment_name = experiment_name
        self.load_run = load_run
        self.checkpoint = checkpoint
        self.device = torch.device(device)

        # Expected path:
        # logs/rsl_rl/<experiment_name>/<load_run>/<checkpoint>
        if os.path.isabs(checkpoint):
            load_path = checkpoint
        else:
            load_path = os.path.abspath(
                os.path.join(
                    "logs",
                    "rsl_rl",
                    experiment_name,
                    str(load_run),
                    checkpoint,
                )
            )

        if not os.path.exists(load_path):
            raise FileNotFoundError(
                f"[LLPolicyWrapper] Exported low-level policy not found:\n"
                f"  {load_path}\n\n"
                f"Expected something like:\n"
                f"  logs/rsl_rl/{experiment_name}/{load_run}/exported/policy.pt"
            )

        print(f"[LLPolicyWrapper] task: {task}")
        print(f"[LLPolicyWrapper] loading exported RSL-RL policy from: {load_path}")

        # RSL-RL exported policy.pt is usually TorchScript.
        self.policy = torch.jit.load(load_path, map_location=self.device)
        self.policy.to(self.device)
        self.policy.eval()

        for param in self.policy.parameters():
            param.requires_grad = False

        print("[LLPolicyWrapper] exported low-level policy loaded and frozen.")

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference.

        Args:
            obs: Low-level observation tensor, shape [num_envs, num_obs].

        Returns:
            Low-level action tensor, shape [num_envs, num_actions].
        """
        obs = obs.to(self.device, dtype=torch.float32)

        with torch.no_grad():
            action = self.policy(obs)

            # Some exported modules may return tuple/list.
            if isinstance(action, (tuple, list)):
                action = action[0]

            return action