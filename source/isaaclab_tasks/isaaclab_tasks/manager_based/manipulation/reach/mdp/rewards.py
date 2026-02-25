# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul, quat_from_euler_xyz, euler_xyz_from_quat, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    Note: pose command is sampled in the asset's body frame.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    Note: pose command is sampled in the asset's body frame.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def position_command_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    err2 = torch.sum((curr_pos_w - des_pos_w) ** 2, dim=1)
    return torch.exp(-err2 / (std**2))


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    Note: pose command is sampled in the asset's body frame.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)


# helper function for level base pose command error
def _level_base_pose_w(asset: RigidObject) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (pos_w, quat_w) of LEVEL base (LB) for each env.
    pos_w: (N,3) base position in world (no projection)
    quat_w: (N,4) yaw-only quaternion (wxyz)
    """
    base_pos_w = asset.data.root_pos_w  # (N,3)
    base_quat_w = asset.data.root_quat_w  # (N,4) wxyz

    # yaw-only orientation: roll/pitch = 0
    roll, pitch, yaw = euler_xyz_from_quat(base_quat_w)
    zeros = torch.zeros_like(yaw)
    lb_quat_w = quat_from_euler_xyz(zeros, zeros, yaw)  # (N,4) wxyz

    # origin = base origin (no projection)
    lb_pos_w = base_pos_w
    return lb_pos_w, lb_quat_w


def keypoints_command_error_exp_lb_delayed(
    env,
    command_name: str,
    std: float,
    asset_cfg,
    kp_dx: float = 0.30,
    kp_dz: float = 0.30,
    w0: float = 1.0,
    w1: float = 1.0,
    w2: float = 1.0,
    std0: float | None = None,
    std1: float | None = None,
    std2: float | None = None,
    track_window_s: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Delayed tracking reward (LB version): only active in last 'track_window_s' seconds of each command cycle.

    Command is assumed in LB frame (N,9) and is transformed LB->World using base yaw-only transform.
    """
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)  # (N,9)

    if command.shape[-1] != 9:
        raise ValueError(
            f"[keypoints_command_error_exp_lb_delayed] Expected command dim 9, got {tuple(command.shape)}"
        )

    # delayed mask from CommandTerm.time_left 
    cmd_term = env.command_manager.get_term(command_name)
    time_left = cmd_term.time_left  # (N,)
    in_window = time_left <= float(track_window_s)

    # average over last window (dt is multiplied outside)
    window_scale = 1.0 / max(float(track_window_s), 1e-6)

    # desired keypoints LB -> world 
    kp0_lb = command[:, 0:3]
    kp1_lb = command[:, 3:6]
    kp2_lb = command[:, 6:9]

    lb_pos_w, lb_quat_w = _level_base_pose_w(asset)
    kp0_des_w, _ = combine_frame_transforms(lb_pos_w, lb_quat_w, kp0_lb)
    kp1_des_w, _ = combine_frame_transforms(lb_pos_w, lb_quat_w, kp1_lb)
    kp2_des_w, _ = combine_frame_transforms(lb_pos_w, lb_quat_w, kp2_lb)

    # current EE keypoints (world) 
    body_id = asset_cfg.body_ids[0]
    if isinstance(body_id, (list, tuple)):
        body_id = body_id[0]
    body_id = int(body_id)

    ee_pos_w = asset.data.body_pos_w[:, body_id, :]      # (N,3)
    ee_quat_w = asset.data.body_quat_w[:, body_id, :]    # (N,4) wxyz

    off_x = ee_pos_w.new_tensor([kp_dx, 0.0, 0.0]).unsqueeze(0).expand(ee_pos_w.shape[0], 3)
    off_z = ee_pos_w.new_tensor([0.0, 0.0, kp_dz]).unsqueeze(0).expand(ee_pos_w.shape[0], 3)

    kp0_cur_w = ee_pos_w
    kp1_cur_w = ee_pos_w + quat_apply(ee_quat_w, off_x)
    kp2_cur_w = ee_pos_w + quat_apply(ee_quat_w, off_z)

    # per-kp exp kernels 
    e0_2 = torch.sum((kp0_cur_w - kp0_des_w) ** 2, dim=1)
    e1_2 = torch.sum((kp1_cur_w - kp1_des_w) ** 2, dim=1)
    e2_2 = torch.sum((kp2_cur_w - kp2_des_w) ** 2, dim=1)

    s0 = float(std if std0 is None else std0)
    s1 = float(std if std1 is None else std1)
    s2 = float(std if std2 is None else std2)

    s0 = max(s0, 1e-6)
    s1 = max(s1, 1e-6)
    s2 = max(s2, 1e-6)

    r0 = torch.exp(-e0_2 / (s0 * s0))
    r1 = torch.exp(-e1_2 / (s1 * s1))
    r2 = torch.exp(-e2_2 / (s2 * s2))

    wsum = float(w0 + w1 + w2)
    if wsum < eps:
        r = (r0 + r1 + r2) / 3.0
    else:
        r = (float(w0) * r0 + float(w1) * r1 + float(w2) * r2) / wsum

    # delayed + window average
    r = torch.where(in_window, r * window_scale, torch.zeros_like(r))
    return r


def keypoints_command_progress_lb_robust(
    env,
    command_name: str,
    asset_cfg,
    kp_dx: float = 0.30,
    kp_dz: float = 0.30,
    active_window_s: float = 6.0,
    pos_clip: float = 0.20,
    neg_clip: float = 0.05,
    w_overshoot: float = 0.5,
    eps: float = 1e-6,
):
    """
    Robust progress reward (LB frame).

    Key ideas:
    - reward = per-step improvement: max(d_prev - d_now, 0)
    - penalize oscillation / overshoot: max(d_now - d_prev, 0)
    - works with moving targets and long command cycles
    """

    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)  # (N,9)
    if command.shape[-1] != 9:
        raise ValueError(
            f"[keypoints_command_progress_lb_robust] Expected command dim 9, got {tuple(command.shape)}"
        )

    cmd_term = env.command_manager.get_term(command_name)
    N = env.num_envs
    device = env.device

    # state buffers (per env)
    if not hasattr(cmd_term, "_prog_prev_d"):
        cmd_term._prog_prev_d = torch.zeros((N, 3), device=device)
        cmd_term._prog_prev_valid = torch.zeros((N,), dtype=torch.bool, device=device)
        cmd_term._prog_elapsed_s = torch.zeros((N,), device=device)
        cmd_term._prog_last_counter = torch.full((N,), -1, device=device, dtype=torch.long)

    # reset at command change
    cc = cmd_term.command_counter
    changed = cc != cmd_term._prog_last_counter
    if torch.any(changed):
        cmd_term._prog_prev_valid[changed] = False
        cmd_term._prog_elapsed_s[changed] = 0.0
        cmd_term._prog_last_counter[changed] = cc[changed]

    cmd_term._prog_elapsed_s += float(env.step_dt)
    active_mask = cmd_term._prog_elapsed_s <= float(active_window_s)

    # desired keypoints (LB -> world)
    kp0_lb = command[:, 0:3]
    kp1_lb = command[:, 3:6]
    kp2_lb = command[:, 6:9]

    lb_pos_w, lb_quat_w = _level_base_pose_w(asset)
    kp0_des_w, _ = combine_frame_transforms(lb_pos_w, lb_quat_w, kp0_lb)
    kp1_des_w, _ = combine_frame_transforms(lb_pos_w, lb_quat_w, kp1_lb)
    kp2_des_w, _ = combine_frame_transforms(lb_pos_w, lb_quat_w, kp2_lb)

    # current EE keypoints (world)
    body_id = asset_cfg.body_ids[0]
    if isinstance(body_id, (list, tuple)):
        body_id = body_id[0]
    body_id = int(body_id)

    ee_pos_w = asset.data.body_pos_w[:, body_id, :]
    ee_quat_w = asset.data.body_quat_w[:, body_id, :]

    off_x = ee_pos_w.new_tensor([kp_dx, 0.0, 0.0]).unsqueeze(0).expand(N, 3)
    off_z = ee_pos_w.new_tensor([0.0, 0.0, kp_dz]).unsqueeze(0).expand(N, 3)

    kp0_cur_w = ee_pos_w
    kp1_cur_w = ee_pos_w + quat_apply(ee_quat_w, off_x)
    kp2_cur_w = ee_pos_w + quat_apply(ee_quat_w, off_z)

    # distances
    d0 = torch.linalg.norm(kp0_cur_w - kp0_des_w, dim=1)
    d1 = torch.linalg.norm(kp1_cur_w - kp1_des_w, dim=1)
    d2 = torch.linalg.norm(kp2_cur_w - kp2_des_w, dim=1)
    d_now = torch.stack([d0, d1, d2], dim=1)  # (N,3)

    # per-step progress
    r = torch.zeros((N,), device=device)

    valid = cmd_term._prog_prev_valid & active_mask
    if torch.any(valid):
        d_prev = cmd_term._prog_prev_d[valid]
        d_cur = d_now[valid]

        improvement = torch.clamp(d_prev - d_cur, min=0.0)
        overshoot = torch.clamp(d_cur - d_prev, min=0.0)

        r_pos = torch.clamp(improvement.mean(dim=1), max=pos_clip)
        r_neg = torch.clamp(overshoot.mean(dim=1), max=neg_clip)

        r[valid] = r_pos - float(w_overshoot) * r_neg

    # update buffer
    cmd_term._prog_prev_d[active_mask] = d_now[active_mask]
    cmd_term._prog_prev_valid[active_mask] = True

    return torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def keypoints_command_tracking_finetune_lb(
    env,
    command_name: str,
    asset_cfg,
    kp_dx: float = 0.30,
    kp_dz: float = 0.30,
    eps: float = 0.20,          # only active within this error radius
    w0: float = 1.0,
    w1: float = 1.0,
    w2: float = 1.0,
) -> torch.Tensor:
    """
    Fine-tuning tracking reward for EE keypoints.
    Active only in small-error regime. Linear, monotonic, low oscillation.
    """

    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)  # (N,9)
    N = env.num_envs

    # --- desired keypoints: LB -> world ---
    kp0_lb = command[:, 0:3]
    kp1_lb = command[:, 3:6]
    kp2_lb = command[:, 6:9]

    lb_pos_w, lb_quat_w = _level_base_pose_w(asset)
    kp0_des_w, _ = combine_frame_transforms(lb_pos_w, lb_quat_w, kp0_lb)
    kp1_des_w, _ = combine_frame_transforms(lb_pos_w, lb_quat_w, kp1_lb)
    kp2_des_w, _ = combine_frame_transforms(lb_pos_w, lb_quat_w, kp2_lb)

    # --- current EE keypoints ---
    body_id = asset_cfg.body_ids[0]
    if isinstance(body_id, (list, tuple)):
        body_id = body_id[0]
    body_id = int(body_id)

    ee_pos_w = asset.data.body_pos_w[:, body_id, :]
    ee_quat_w = asset.data.body_quat_w[:, body_id, :]

    off_x = ee_pos_w.new_tensor([kp_dx, 0.0, 0.0]).unsqueeze(0).expand(N, 3)
    off_z = ee_pos_w.new_tensor([0.0, 0.0, kp_dz]).unsqueeze(0).expand(N, 3)

    kp0_cur_w = ee_pos_w
    kp1_cur_w = ee_pos_w + quat_apply(ee_quat_w, off_x)
    kp2_cur_w = ee_pos_w + quat_apply(ee_quat_w, off_z)

    # --- distances ---
    d0 = torch.linalg.norm(kp0_cur_w - kp0_des_w, dim=1)
    d1 = torch.linalg.norm(kp1_cur_w - kp1_des_w, dim=1)
    d2 = torch.linalg.norm(kp2_cur_w - kp2_des_w, dim=1)

    # --- clipped linear reward ---
    r0 = torch.clamp(1.0 - d0 / eps, min=0.0)
    r1 = torch.clamp(1.0 - d1 / eps, min=0.0)
    r2 = torch.clamp(1.0 - d2 / eps, min=0.0)

    r = (w0 * r0 + w1 * r1 + w2 * r2) / (w0 + w1 + w2)
    return r
