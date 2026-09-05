# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import os

import torch
import torch.nn.functional as F


def export_policy_as_jit(
    policy: object,
    normalizer: object | None,
    path: str,
    filename: str = "policy.pt",
):
    """Export a Gaussian or Beta policy into a TorchScript file.

    The exported module includes actor observation normalization and returns
    deterministic environment-space actions. Beta policies use the configured
    inference mode, normally the distribution mean.
    """
    policy_exporter = _TorchPolicyExporter(policy, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(
    policy: object,
    path: str,
    normalizer: object | None = None,
    filename: str = "policy.onnx",
    verbose: bool = False,
):
    """Export a Gaussian or Beta policy into an ONNX file."""
    os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(policy, normalizer, verbose)
    policy_exporter.export(path, filename)


class _PolicyDistributionDecoder:
    """Distribution-specific deterministic actor-output decoding."""

    def _configure_distribution(self, policy: object) -> None:
        distribution_type = str(getattr(policy, "distribution_type", "gaussian")).lower()
        if distribution_type not in ("gaussian", "beta"):
            raise ValueError(
                f"Unsupported distribution_type '{distribution_type}'. "
                "Expected 'gaussian' or 'beta'."
            )

        self.is_beta = distribution_type == "beta"
        self.state_dependent_std = bool(getattr(policy, "state_dependent_std", False))
        self.beta_min_concentration = float(getattr(policy, "beta_min_concentration", 1.0))
        self.beta_eps = float(getattr(policy, "beta_eps", 1.0e-6))
        beta_inference_mode = str(getattr(policy, "beta_inference_mode", "mean")).lower()
        if beta_inference_mode not in ("mean", "mode"):
            raise ValueError(
                f"Unsupported beta_inference_mode '{beta_inference_mode}'."
            )
        self.use_beta_mode = beta_inference_mode == "mode"

        if self.is_beta and self.state_dependent_std:
            raise ValueError(
                "Beta export requires state_dependent_std=False."
            )

    def _decode_actor_output(self, actor_output: torch.Tensor) -> torch.Tensor:
        if self.is_beta:
            # Actor output: [..., 2, num_actions].
            raw_alpha = actor_output.select(dim=-2, index=0)
            raw_beta = actor_output.select(dim=-2, index=1)

            alpha = (
                F.softplus(raw_alpha)
                + self.beta_min_concentration
                + self.beta_eps
            )
            beta = (
                F.softplus(raw_beta)
                + self.beta_min_concentration
                + self.beta_eps
            )

            unit_mean = alpha / (alpha + beta)
            if self.use_beta_mode:
                denominator = alpha + beta - 2.0
                unit_mode = (alpha - 1.0) / torch.clamp(
                    denominator,
                    min=self.beta_eps,
                )
                valid_mode = (
                    (alpha > 1.0)
                    & (beta > 1.0)
                    & (denominator > 0.1)
                )
                unit_action = torch.where(valid_mode, unit_mode, unit_mean)
            else:
                unit_action = unit_mean

            unit_action = torch.clamp(
                unit_action,
                min=self.beta_eps,
                max=1.0 - self.beta_eps,
            )
            return 2.0 * unit_action - 1.0

        # Standard Gaussian deterministic inference returns its mean. For a
        # state-dependent-std actor, output index zero contains the mean.
        if self.state_dependent_std:
            return actor_output.select(dim=-2, index=0)
        return actor_output


class _TorchPolicyExporter(torch.nn.Module, _PolicyDistributionDecoder):
    """Exporter of actor-critic policies into TorchScript."""

    def __init__(self, policy: object, normalizer: object | None = None):
        super().__init__()
        self.is_recurrent = bool(policy.is_recurrent)
        self._configure_distribution(policy)

        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
            # Distillation students are treated as direct deterministic actors.
            self.is_beta = False
            self.state_dependent_std = False
            self.use_beta_mode = False
        else:
            raise ValueError("Policy does not have an actor/student module.")

        self.normalizer = (
            copy.deepcopy(normalizer)
            if normalizer is not None
            else torch.nn.Identity()
        )

        self.actor.eval()
        self.normalizer.eval()

        if self.is_recurrent:
            self.rnn.cpu()
            self.rnn.eval()
            self.rnn_type = type(self.rnn).__name__.lower()
            self.register_buffer(
                "hidden_state",
                torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size),
            )
            if self.rnn_type == "lstm":
                self.register_buffer(
                    "cell_state",
                    torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size),
                )
                self.forward = self.forward_lstm
                self.reset = self.reset_memory
            elif self.rnn_type == "gru":
                self.forward = self.forward_gru
                self.reset = self.reset_memory
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")

    def forward_lstm(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalizer(x)
        x, (h, c) = self.rnn(
            x.unsqueeze(0),
            (self.hidden_state, self.cell_state),
        )
        self.hidden_state[:] = h
        self.cell_state[:] = c
        actor_output = self.actor(x.squeeze(0))
        return self._decode_actor_output(actor_output)

    def forward_gru(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalizer(x)
        x, h = self.rnn(x.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h
        actor_output = self.actor(x.squeeze(0))
        return self._decode_actor_output(actor_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        actor_output = self.actor(self.normalizer(x))
        return self._decode_actor_output(actor_output)

    @torch.jit.export
    def reset(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        if hasattr(self, "cell_state"):
            self.cell_state[:] = 0.0

    def export(self, path: str, filename: str) -> None:
        os.makedirs(path, exist_ok=True)
        export_path = os.path.join(path, filename)
        self.to("cpu")
        self.eval()
        scripted_module = torch.jit.script(self)
        scripted_module.save(export_path)


class _OnnxPolicyExporter(torch.nn.Module, _PolicyDistributionDecoder):
    """Exporter of actor-critic policies into ONNX."""

    def __init__(
        self,
        policy: object,
        normalizer: object | None = None,
        verbose: bool = False,
    ):
        super().__init__()
        self.verbose = verbose
        self.is_recurrent = bool(policy.is_recurrent)
        self._configure_distribution(policy)

        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
            self.is_beta = False
            self.state_dependent_std = False
            self.use_beta_mode = False
        else:
            raise ValueError("Policy does not have an actor/student module.")

        self.normalizer = (
            copy.deepcopy(normalizer)
            if normalizer is not None
            else torch.nn.Identity()
        )

        self.actor.eval()
        self.normalizer.eval()

        if self.is_recurrent:
            self.rnn.cpu()
            self.rnn.eval()
            self.rnn_type = type(self.rnn).__name__.lower()
            if self.rnn_type == "lstm":
                self.forward = self.forward_lstm
            elif self.rnn_type == "gru":
                self.forward = self.forward_gru
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")

    def forward_lstm(
        self,
        x_in: torch.Tensor,
        h_in: torch.Tensor,
        c_in: torch.Tensor,
    ):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        actor_output = self.actor(x.squeeze(0))
        actions = self._decode_actor_output(actor_output)
        return actions, h, c

    def forward_gru(
        self,
        x_in: torch.Tensor,
        h_in: torch.Tensor,
    ):
        x_in = self.normalizer(x_in)
        x, h = self.rnn(x_in.unsqueeze(0), h_in)
        actor_output = self.actor(x.squeeze(0))
        actions = self._decode_actor_output(actor_output)
        return actions, h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        actor_output = self.actor(self.normalizer(x))
        return self._decode_actor_output(actor_output)

    @staticmethod
    def _infer_actor_input_size(actor: torch.nn.Module) -> int:
        for module in actor.modules():
            if isinstance(module, torch.nn.Linear):
                return int(module.in_features)
        raise RuntimeError("Could not infer actor input size from the exported actor.")

    def export(self, path: str, filename: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.to("cpu")
        self.eval()
        output_path = os.path.join(path, filename)

        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(
                self.rnn.num_layers,
                1,
                self.rnn.hidden_size,
            )

            if self.rnn_type == "lstm":
                c_in = torch.zeros(
                    self.rnn.num_layers,
                    1,
                    self.rnn.hidden_size,
                )
                torch.onnx.export(
                    self,
                    (obs, h_in, c_in),
                    output_path,
                    export_params=True,
                    opset_version=13,
                    verbose=self.verbose,
                    input_names=["obs", "h_in", "c_in"],
                    output_names=["actions", "h_out", "c_out"],
                    dynamic_axes={},
                )
            elif self.rnn_type == "gru":
                torch.onnx.export(
                    self,
                    (obs, h_in),
                    output_path,
                    export_params=True,
                    opset_version=13,
                    verbose=self.verbose,
                    input_names=["obs", "h_in"],
                    output_names=["actions", "h_out"],
                    dynamic_axes={},
                )
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        else:
            obs = torch.zeros(1, self._infer_actor_input_size(self.actor))
            torch.onnx.export(
                self,
                obs,
                output_path,
                export_params=True,
                opset_version=13,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )