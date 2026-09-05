# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils


def transform_points_w_to_plb(
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    points_w: torch.Tensor,
    ground_z: float = 0.0,
) -> torch.Tensor:
    """Transform world points to projected roll-pitch-invariant base frame.

    PLB frame convention:
        origin.x = base.x
        origin.y = base.y
        origin.z = ground_z
        rotation = base yaw only

    Args:
        root_pos_w: Root position in world frame, shape [num_envs, 3].
        root_quat_w: Root orientation in world frame, shape [num_envs, 4], wxyz.
        points_w: Points in world frame, shape [num_envs, num_points, 3].
        ground_z: World-frame z coordinate used as PLB z-origin.

    Returns:
        Points in PLB frame, shape [num_envs, num_points, 3].
    """
    num_envs, num_points, _ = points_w.shape

    yaw_quat = math_utils.yaw_quat(root_quat_w)
    yaw_quat_inv = math_utils.quat_inv(yaw_quat)

    origin_w = root_pos_w.clone()
    origin_w[:, 2] = ground_z

    rel_w = points_w - origin_w.unsqueeze(1)

    rel_w_flat = rel_w.reshape(num_envs * num_points, 3)
    yaw_inv_flat = yaw_quat_inv.unsqueeze(1).expand(-1, num_points, -1).reshape(
        num_envs * num_points,
        4,
    )

    points_plb = math_utils.quat_apply(yaw_inv_flat, rel_w_flat)
    return points_plb.reshape(num_envs, num_points, 3)


def compute_ee_keypoints_plb(
    robot,
    ee_body_id,
    kp_dx: float,
    kp_dz: float,
    ground_z: float = 0.0,
) -> torch.Tensor:
    """Compute current EE keypoints in ground-based PLB frame.

    Keypoints:
        kp0 = EE position
        kp1 = kp0 + kp_dx * EE local x-axis
        kp2 = kp0 + kp_dz * EE local z-axis

    Args:
        robot: Isaac Lab Articulation.
        ee_body_id: Body index for the end-effector body.
        kp_dx: Offset length along EE local x-axis.
        kp_dz: Offset length along EE local z-axis.
        ground_z: World-frame z coordinate used as PLB z-origin.

    Returns:
        EE keypoints in PLB frame, shape [num_envs, 9], ordered [kp0, kp1, kp2].
    """
    if isinstance(ee_body_id, torch.Tensor):
        ee_body_id = int(ee_body_id.flatten()[0].item())
    elif isinstance(ee_body_id, (list, tuple)):
        ee_body_id = ee_body_id[0]
        if isinstance(ee_body_id, torch.Tensor):
            ee_body_id = int(ee_body_id.item())
        else:
            ee_body_id = int(ee_body_id)
    else:
        ee_body_id = int(ee_body_id)

    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    ee_pos_w = robot.data.body_pos_w[:, ee_body_id]
    ee_quat_w = robot.data.body_quat_w[:, ee_body_id]

    num_envs = ee_pos_w.shape[0]
    device = ee_pos_w.device

    x_axis_local = torch.tensor(
        [1.0, 0.0, 0.0],
        dtype=torch.float32,
        device=device,
    ).repeat(num_envs, 1)

    z_axis_local = torch.tensor(
        [0.0, 0.0, 1.0],
        dtype=torch.float32,
        device=device,
    ).repeat(num_envs, 1)

    x_axis_w = math_utils.quat_apply(ee_quat_w, x_axis_local)
    z_axis_w = math_utils.quat_apply(ee_quat_w, z_axis_local)

    kp0_w = ee_pos_w
    kp1_w = ee_pos_w + kp_dx * x_axis_w
    kp2_w = ee_pos_w + kp_dz * z_axis_w

    points_w = torch.stack([kp0_w, kp1_w, kp2_w], dim=1)

    points_plb = transform_points_w_to_plb(
        root_pos_w=root_pos_w,
        root_quat_w=root_quat_w,
        points_w=points_w,
        ground_z=ground_z,
    )

    return points_plb.reshape(num_envs, 9)