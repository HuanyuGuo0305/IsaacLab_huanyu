# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


def build_keypoints_from_kp0_yaw_pitch_plb(
    kp0: torch.Tensor,
    yaw: torch.Tensor,
    pitch: torch.Tensor,
    kp_dx: float = 0.30,
    kp_dz: float = 0.30,
    eps: float = 1.0e-6,
    *,
    roll: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build an EE keypoint command in PLB frame from position and orientation.

    Keypoint convention:
        kp0 = end-effector position in PLB frame
        kp1 = kp0 + kp_dx * EE local x-axis
        kp2 = kp0 + kp_dz * EE local z-axis

    Orientation convention:
        - yaw controls the heading of the EE local x-axis in the PLB xy plane.
        - pitch controls the elevation of the EE local x-axis.
        - roll rotates the EE local z-axis about the EE local x-axis.

    The construction is intentionally backward compatible with the previous
    roll-free adapter: when ``roll`` is omitted or identically zero, the
    generated kp0/kp1/kp2 commands match the prior yaw-pitch parameterization.

    Rotation semantics:
        1) Construct EE local x-axis from yaw and pitch.
        2) Construct a zero-roll EE local z-axis as the most-upward direction
           orthogonal to local x.
        3) Apply roll around local x using the right-hand rule.

    Args:
        kp0: Tensor of shape [num_envs, 3], EE position in PLB frame.
        yaw: Tensor of shape [num_envs], yaw angle in radians.
        pitch: Tensor of shape [num_envs], pitch/elevation angle in radians.
            Positive pitch points the EE local x-axis upward in PLB z.
        kp_dx: Distance from kp0 to kp1 along EE local x-axis.
        kp_dz: Distance from kp0 to kp2 along EE local z-axis.
        eps: Small value for numerical stability.
        roll: Optional tensor of shape [num_envs], roll angle in radians.
            Positive roll follows the right-hand rule about EE local +x.
            If omitted, zero roll is used.

    Returns:
        Tensor of shape [num_envs, 9], ordered as:
            [kp0_x, kp0_y, kp0_z,
             kp1_x, kp1_y, kp1_z,
             kp2_x, kp2_y, kp2_z]
    """
    if kp0.ndim != 2 or kp0.shape[-1] != 3:
        raise ValueError(f"Expected kp0 shape [num_envs, 3], got {tuple(kp0.shape)}")

    batch_size = kp0.shape[0]

    if yaw.ndim != 1:
        yaw = yaw.reshape(-1)
    if pitch.ndim != 1:
        pitch = pitch.reshape(-1)

    if yaw.shape[0] != batch_size:
        raise ValueError(
            f"yaw batch size mismatch: expected {batch_size}, got {yaw.shape[0]}"
        )
    if pitch.shape[0] != batch_size:
        raise ValueError(
            f"pitch batch size mismatch: expected {batch_size}, got {pitch.shape[0]}"
        )

    if roll is None:
        roll = torch.zeros(batch_size, device=kp0.device, dtype=kp0.dtype)
    elif roll.ndim != 1:
        roll = roll.reshape(-1)

    if roll.shape[0] != batch_size:
        raise ValueError(
            f"roll batch size mismatch: expected {batch_size}, got {roll.shape[0]}"
        )

    device = kp0.device
    dtype = kp0.dtype
    yaw = yaw.to(device=device, dtype=dtype)
    pitch = pitch.to(device=device, dtype=dtype)
    roll = roll.to(device=device, dtype=dtype)

    # EE local x-axis in PLB. This is unchanged from the previous adapter.
    cos_pitch = torch.cos(pitch)
    v_x = torch.stack(
        [
            cos_pitch * torch.cos(yaw),
            cos_pitch * torch.sin(yaw),
            torch.sin(pitch),
        ],
        dim=-1,
    )
    v_x = v_x / torch.clamp(torch.norm(v_x, dim=-1, keepdim=True), min=eps)

    # Zero-roll EE local z-axis: the most-upward vector orthogonal to v_x.
    # This exactly preserves the former yaw-pitch-only command at roll == 0.
    up = torch.zeros_like(v_x)
    up[:, 2] = 1.0

    up_dot_v_x = torch.sum(up * v_x, dim=-1, keepdim=True)
    v_z_zero = up - up_dot_v_x * v_x
    v_z_zero_norm = torch.norm(v_z_zero, dim=-1, keepdim=True)

    # Safe zero-roll fallback for the singular case where local x is nearly
    # parallel to PLB +Z/-Z. The current configured pitch range avoids it.
    fallback_z = torch.zeros_like(v_x)
    fallback_z[:, 0] = -torch.sin(yaw)
    fallback_z[:, 1] = torch.cos(yaw)
    fallback_z[:, 2] = 0.0
    fallback_z = fallback_z / torch.clamp(
        torch.norm(fallback_z, dim=-1, keepdim=True), min=eps
    )

    v_z_zero = torch.where(
        v_z_zero_norm > eps,
        v_z_zero / torch.clamp(v_z_zero_norm, min=eps),
        fallback_z,
    )

    # Complete the zero-roll right-handed frame: x cross y = z, so y = z cross x.
    v_y_zero = torch.linalg.cross(v_z_zero, v_x, dim=-1)
    v_y_zero = v_y_zero / torch.clamp(
        torch.norm(v_y_zero, dim=-1, keepdim=True), min=eps
    )

    # Apply local roll about +x. For a right-handed local frame:
    #   z(roll) = cos(roll) * z(0) - sin(roll) * y(0)
    cos_roll = torch.cos(roll).unsqueeze(-1)
    sin_roll = torch.sin(roll).unsqueeze(-1)
    v_z = cos_roll * v_z_zero - sin_roll * v_y_zero
    v_z = v_z / torch.clamp(torch.norm(v_z, dim=-1, keepdim=True), min=eps)

    kp1 = kp0 + kp_dx * v_x
    kp2 = kp0 + kp_dz * v_z

    return torch.cat([kp0, kp1, kp2], dim=-1)


def split_keypoints(kp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split flattened keypoints [num_envs, 9] into kp0, kp1, kp2.

    Args:
        kp: Tensor of shape [num_envs, 9].

    Returns:
        kp0, kp1, kp2, each with shape [num_envs, 3].
    """
    if kp.ndim != 2 or kp.shape[-1] != 9:
        raise ValueError(f"Expected keypoints shape [num_envs, 9], got {tuple(kp.shape)}")

    return kp[:, 0:3], kp[:, 3:6], kp[:, 6:9]


def keypoint_axes_from_flat(
    kp: torch.Tensor,
    kp_dx: float = 0.30,
    kp_dz: float = 0.30,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover approximate EE local x/z axes from flattened keypoints.

    Args:
        kp: Tensor of shape [num_envs, 9], ordered [kp0, kp1, kp2].
        kp_dx: Expected kp0-to-kp1 distance.
        kp_dz: Expected kp0-to-kp2 distance.
        eps: Small value for numerical stability.

    Returns:
        v1, v2:
            v1 = normalized kp0->kp1 direction, shape [num_envs, 3]
            v2 = normalized kp0->kp2 direction, shape [num_envs, 3]
    """
    kp0, kp1, kp2 = split_keypoints(kp)

    v1 = (kp1 - kp0) / max(kp_dx, eps)
    v2 = (kp2 - kp0) / max(kp_dz, eps)

    v1 = v1 / torch.clamp(torch.norm(v1, dim=-1, keepdim=True), min=eps)
    v2 = v2 / torch.clamp(torch.norm(v2, dim=-1, keepdim=True), min=eps)

    return v1, v2
