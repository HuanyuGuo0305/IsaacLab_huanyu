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

"""
# helper function for level base pose command error
"""

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


def _phase_gate_from_command(
    env,
    command_name: str,
    start_ratio: float,
    end_ratio: float,
    kind: str = "smoothstep",
) -> torch.Tensor:
    """Return phase gate in [0,1] based on command cycle progress.

    phase = 0 at command switch, phase = 1 at command end.
    gate = 0 before start_ratio, 1 after end_ratio, smooth in between.
    """
    cmd_term = env.command_manager.get_term(command_name)
    time_left = cmd_term.time_left  # (N,)

    if hasattr(cmd_term, "cfg") and hasattr(cmd_term.cfg, "resampling_time_range"):
        command_duration = float(cmd_term.cfg.resampling_time_range[0])
    else:
        command_duration = 1.0

    phase = 1.0 - torch.clamp(time_left / max(command_duration, 1e-6), 0.0, 1.0)
    u = (phase - float(start_ratio)) / max(float(end_ratio - start_ratio), 1e-6)
    u = torch.clamp(u, 0.0, 1.0)

    if kind == "smoothstep":
        # 3u^2 - 2u^3
        g = u * u * (3.0 - 2.0 * u)
    elif kind == "smootherstep":
        # 10u^3 - 15u^4 + 6u^5
        g = u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)
    elif kind == "linear":
        g = u
    else:
        raise ValueError(f"Unsupported gate kind: {kind}")

    return g


def _ee_kp_errors_lb(
    env,
    command_name: str,
    asset_cfg,
    kp_dx: float = 0.30,
    kp_dz: float = 0.30,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-keypoint tracking errors in world frame.

    Command is given in LB frame, current EE keypoints are computed in world frame.
    """
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)  # (N, 9)

    if command.shape[-1] != 9:
        raise ValueError(f"[_ee_kp_errors_lb] Expected command dim 9, got {tuple(command.shape)}")

    # desired keypoints in LB
    kp0_lb = command[:, 0:3]
    kp1_lb = command[:, 3:6]
    kp2_lb = command[:, 6:9]

    # LB -> world
    lb_pos_w, lb_quat_w = _level_base_pose_w(asset)
    kp0_des_w, _ = combine_frame_transforms(lb_pos_w, lb_quat_w, kp0_lb)
    kp1_des_w, _ = combine_frame_transforms(lb_pos_w, lb_quat_w, kp1_lb)
    kp2_des_w, _ = combine_frame_transforms(lb_pos_w, lb_quat_w, kp2_lb)

    # current EE keypoints in world
    body_id = asset_cfg.body_ids[0]
    if isinstance(body_id, (list, tuple)):
        body_id = body_id[0]
    body_id = int(body_id)

    ee_pos_w = asset.data.body_pos_w[:, body_id, :]
    ee_quat_w = asset.data.body_quat_w[:, body_id, :]

    off_x = ee_pos_w.new_tensor([kp_dx, 0.0, 0.0]).unsqueeze(0).expand(ee_pos_w.shape[0], 3)
    off_z = ee_pos_w.new_tensor([0.0, 0.0, kp_dz]).unsqueeze(0).expand(ee_pos_w.shape[0], 3)

    kp0_cur_w = ee_pos_w
    kp1_cur_w = ee_pos_w + quat_apply(ee_quat_w, off_x)
    kp2_cur_w = ee_pos_w + quat_apply(ee_quat_w, off_z)

    e0 = torch.linalg.norm(kp0_cur_w - kp0_des_w, dim=1)
    e1 = torch.linalg.norm(kp1_cur_w - kp1_des_w, dim=1)
    e2 = torch.linalg.norm(kp2_cur_w - kp2_des_w, dim=1)

    return e0, e1, e2


"""
custom rewards for manipulation MDPs
"""


def ee_kp_tracking_exp_lb(
    env,
    command_name: str,
    asset_cfg,
    std: float = 0.20,
    kp_dx: float = 0.30,
    kp_dz: float = 0.30,
    w0: float = 1.2,
    w1: float = 1.0,
    w2: float = 1.0,
    std0: float | None = None,
    std1: float | None = None,
    std2: float | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """LB-frame end-effector keypoint tracking reward with standard exponential kernel.

    Reward:
        r_i = exp(-||e_i||^2 / std_i^2)
        r = weighted average of (r0, r1, r2)

    where:
        - kp0 mainly constrains EE position
        - kp1/kp2 constrain EE orientation through offset keypoints
    """
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)  # (N, 9)

    if command.shape[-1] != 9:
        raise ValueError(
            f"[ee_kp_tracking_exp_lb] Expected command dim 9, got {tuple(command.shape)}"
        )

    # desired keypoints in LB
    kp0_lb = command[:, 0:3]
    kp1_lb = command[:, 3:6]
    kp2_lb = command[:, 6:9]

    # LB -> world
    lb_pos_w, lb_quat_w = _level_base_pose_w(asset)
    kp0_des_w, _ = combine_frame_transforms(lb_pos_w, lb_quat_w, kp0_lb)
    kp1_des_w, _ = combine_frame_transforms(lb_pos_w, lb_quat_w, kp1_lb)
    kp2_des_w, _ = combine_frame_transforms(lb_pos_w, lb_quat_w, kp2_lb)

    # current EE keypoints in world
    body_id = asset_cfg.body_ids[0]
    if isinstance(body_id, (list, tuple)):
        body_id = body_id[0]
    body_id = int(body_id)

    ee_pos_w = asset.data.body_pos_w[:, body_id, :]
    ee_quat_w = asset.data.body_quat_w[:, body_id, :]

    off_x = ee_pos_w.new_tensor([kp_dx, 0.0, 0.0]).unsqueeze(0).expand(ee_pos_w.shape[0], 3)
    off_z = ee_pos_w.new_tensor([0.0, 0.0, kp_dz]).unsqueeze(0).expand(ee_pos_w.shape[0], 3)

    kp0_cur_w = ee_pos_w
    kp1_cur_w = ee_pos_w + quat_apply(ee_quat_w, off_x)
    kp2_cur_w = ee_pos_w + quat_apply(ee_quat_w, off_z)

    # squared tracking errors
    e0_2 = torch.sum((kp0_cur_w - kp0_des_w) ** 2, dim=1)
    e1_2 = torch.sum((kp1_cur_w - kp1_des_w) ** 2, dim=1)
    e2_2 = torch.sum((kp2_cur_w - kp2_des_w) ** 2, dim=1)

    s0 = max(float(std if std0 is None else std0), 1e-6)
    s1 = max(float(std if std1 is None else std1), 1e-6)
    s2 = max(float(std if std2 is None else std2), 1e-6)

    r0 = torch.exp(-e0_2 / (s0 * s0))
    r1 = torch.exp(-e1_2 / (s1 * s1))
    r2 = torch.exp(-e2_2 / (s2 * s2))

    wsum = float(w0 + w1 + w2)
    if wsum < eps:
        return (r0 + r1 + r2) / 3.0
    else:
        return (float(w0) * r0 + float(w1) * r1 + float(w2) * r2) / wsum


def ee_kp_tracking_exp_saturated_lb(
    env,
    command_name: str,
    asset_cfg,
    std: float = 0.50,
    kp_dx: float = 0.30,
    kp_dz: float = 0.30,
    w0: float = 1.2,
    w1: float = 1.0,
    w2: float = 1.0,
    std0: float | None = None,
    std1: float | None = None,
    std2: float | None = None,
    threshold: float = 0.20,
    eps: float = 1e-6,
) -> torch.Tensor:
    """LB-frame saturated coarse keypoint tracking reward.

    Reward:
        r_i = exp(-max(||e_i||, threshold)^2 / std_i^2)

    This means the coarse reward stops increasing once keypoint error is below threshold.
    Fine reward should dominate inside the threshold region.
    """
    e0, e1, e2 = _ee_kp_errors_lb(
        env=env,
        command_name=command_name,
        asset_cfg=asset_cfg,
        kp_dx=kp_dx,
        kp_dz=kp_dz,
    )

    e0_eff = torch.clamp(e0, min=float(threshold))
    e1_eff = torch.clamp(e1, min=float(threshold))
    e2_eff = torch.clamp(e2, min=float(threshold))

    s0 = max(float(std if std0 is None else std0), 1e-6)
    s1 = max(float(std if std1 is None else std1), 1e-6)
    s2 = max(float(std if std2 is None else std2), 1e-6)

    r0 = torch.exp(-(e0_eff * e0_eff) / (s0 * s0))
    r1 = torch.exp(-(e1_eff * e1_eff) / (s1 * s1))
    r2 = torch.exp(-(e2_eff * e2_eff) / (s2 * s2))

    wsum = float(w0 + w1 + w2)
    if wsum < eps:
        return (r0 + r1 + r2) / 3.0
    return (float(w0) * r0 + float(w1) * r1 + float(w2) * r2) / wsum


def ee_kp_tracking_delayed_exp_lb(
    env,
    command_name: str,
    asset_cfg,
    std: float = 0.10,
    kp_dx: float = 0.30,
    kp_dz: float = 0.30,
    w0: float = 1.2,
    w1: float = 1.0,
    w2: float = 1.0,
    std0: float | None = None,
    std1: float | None = None,
    std2: float | None = None,
    gate_start_ratio: float = 0.35,
    gate_end_ratio: float = 0.75,
    gate_kind: str = "smootherstep",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Late-phase gated small-std exp tracking reward in LB frame.

    Early phase: weak / near zero.
    Mid-to-late phase: smoothly turns on.
    Purpose: final convergence, without inducing immediate aggressive chase at command switch.
    """
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)  # (N, 9)

    if command.shape[-1] != 9:
        raise ValueError(
            f"[ee_kp_tracking_late_phase_exp_lb] Expected command dim 9, got {tuple(command.shape)}"
        )

    # desired keypoints in LB
    kp0_lb = command[:, 0:3]
    kp1_lb = command[:, 3:6]
    kp2_lb = command[:, 6:9]

    # LB -> world
    lb_pos_w, lb_quat_w = _level_base_pose_w(asset)
    kp0_des_w, _ = combine_frame_transforms(lb_pos_w, lb_quat_w, kp0_lb)
    kp1_des_w, _ = combine_frame_transforms(lb_pos_w, lb_quat_w, kp1_lb)
    kp2_des_w, _ = combine_frame_transforms(lb_pos_w, lb_quat_w, kp2_lb)

    # current EE keypoints in world
    body_id = asset_cfg.body_ids[0]
    if isinstance(body_id, (list, tuple)):
        body_id = body_id[0]
    body_id = int(body_id)

    ee_pos_w = asset.data.body_pos_w[:, body_id, :]
    ee_quat_w = asset.data.body_quat_w[:, body_id, :]

    off_x = ee_pos_w.new_tensor([kp_dx, 0.0, 0.0]).unsqueeze(0).expand(ee_pos_w.shape[0], 3)
    off_z = ee_pos_w.new_tensor([0.0, 0.0, kp_dz]).unsqueeze(0).expand(ee_pos_w.shape[0], 3)

    kp0_cur_w = ee_pos_w
    kp1_cur_w = ee_pos_w + quat_apply(ee_quat_w, off_x)
    kp2_cur_w = ee_pos_w + quat_apply(ee_quat_w, off_z)

    # squared tracking errors
    e0_2 = torch.sum((kp0_cur_w - kp0_des_w) ** 2, dim=1)
    e1_2 = torch.sum((kp1_cur_w - kp1_des_w) ** 2, dim=1)
    e2_2 = torch.sum((kp2_cur_w - kp2_des_w) ** 2, dim=1)

    s0 = max(float(std if std0 is None else std0), 1e-6)
    s1 = max(float(std if std1 is None else std1), 1e-6)
    s2 = max(float(std if std2 is None else std2), 1e-6)

    r0 = torch.exp(-e0_2 / (s0 * s0))
    r1 = torch.exp(-e1_2 / (s1 * s1))
    r2 = torch.exp(-e2_2 / (s2 * s2))

    wsum = float(w0 + w1 + w2)
    if wsum < eps:
        r_track = (r0 + r1 + r2) / 3.0
    else:
        r_track = (float(w0) * r0 + float(w1) * r1 + float(w2) * r2) / wsum

    late_gate = _phase_gate_from_command(
        env,
        command_name=command_name,
        start_ratio=gate_start_ratio,
        end_ratio=gate_end_ratio,
        kind=gate_kind,
    )

    return late_gate * r_track


def ee_kp_tracking_late_phase_tanh_lb(
    env,
    command_name: str,
    asset_cfg,
    std: float = 0.10,
    kp_dx: float = 0.30,
    kp_dz: float = 0.30,
    w0: float = 1.2,
    w1: float = 1.0,
    w2: float = 1.0,
    std0: float | None = None,
    std1: float | None = None,
    std2: float | None = None,
    gate_start_ratio: float = 0.25,
    gate_end_ratio: float = 0.75,
    gate_kind: str = "smootherstep",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Late-phase gated tanh keypoint tracking reward in LB frame.

    Reward:
        r_i = 1 - tanh(||e_i|| / std_i)
        r = late_gate * weighted_average(r0, r1, r2)

    Compared with small-std exp, tanh is smoother and less sparse.
    """
    e0, e1, e2 = _ee_kp_errors_lb(
        env=env,
        command_name=command_name,
        asset_cfg=asset_cfg,
        kp_dx=kp_dx,
        kp_dz=kp_dz,
    )

    s0 = max(float(std if std0 is None else std0), 1e-6)
    s1 = max(float(std if std1 is None else std1), 1e-6)
    s2 = max(float(std if std2 is None else std2), 1e-6)

    r0 = 1.0 - torch.tanh(e0 / s0)
    r1 = 1.0 - torch.tanh(e1 / s1)
    r2 = 1.0 - torch.tanh(e2 / s2)

    wsum = float(w0 + w1 + w2)
    if wsum < eps:
        r_track = (r0 + r1 + r2) / 3.0
    else:
        r_track = (float(w0) * r0 + float(w1) * r1 + float(w2) * r2) / wsum

    late_gate = _phase_gate_from_command(
        env,
        command_name=command_name,
        start_ratio=gate_start_ratio,
        end_ratio=gate_end_ratio,
        kind=gate_kind,
    )

    return late_gate * r_track


def ee_kp_tracking_sparse_success_lb(
    env,
    command_name: str,
    asset_cfg,
    kp_dx: float = 0.30,
    kp_dz: float = 0.30,
    w0: float = 1.2,
    w1: float = 1.0,
    w2: float = 1.0,
    th1: float = 0.10,
    th2: float = 0.05,
    th3: float = 0.03,
    th4: float = 0.01,
    bonus1: float = 0.2,
    bonus2: float = 0.2,
    bonus3: float = 0.3,
    bonus4: float = 0.3,
    metric: str = "weighted_mean",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Sparse success bonus for LB keypoint tracking.

    If metric='weighted_mean':
        err = weighted mean of keypoint errors.

    If metric='max':
        err = max(e0, e1, e2), stricter because all keypoints must be close.

    Reward:
        +bonus1 if err < th1
        +bonus2 if err < th2
    """
    e0, e1, e2 = _ee_kp_errors_lb(
        env=env,
        command_name=command_name,
        asset_cfg=asset_cfg,
        kp_dx=kp_dx,
        kp_dz=kp_dz,
    )

    if metric == "weighted_mean":
        wsum = max(float(w0 + w1 + w2), eps)
        err = (float(w0) * e0 + float(w1) * e1 + float(w2) * e2) / wsum
    elif metric == "max":
        err = torch.maximum(torch.maximum(e0, e1), e2)
    else:
        raise ValueError(f"[ee_kp_tracking_sparse_success_lb] Unsupported metric: {metric}")

    reward = torch.zeros_like(err)
    reward = reward + float(bonus1) * (err < float(th1)).float()
    reward = reward + float(bonus2) * (err < float(th2)).float()
    reward = reward + float(bonus3) * (err < float(th3)).float()
    reward = reward + float(bonus4) * (err < float(th4)).float()
    return reward


def ee_kp_tracking_exp_saturated_std_schedule_lb(
    env,
    command_name: str,
    asset_cfg,
    std_start: float = 0.80,
    std_end: float = 0.50,
    kp_dx: float = 0.30,
    kp_dz: float = 0.30,
    w0: float = 1.2,
    w1: float = 1.0,
    w2: float = 1.0,
    threshold: float = 0.20,
    gate_start_ratio: float = 0.0,
    gate_end_ratio: float = 0.50,
    gate_kind: str = "smootherstep",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Scheduled saturated coarse LB keypoint tracking reward.

    Behavior:
      - During command phase 0% -> 50%, std smoothly ramps from std_start to std_end.
      - During command phase 50% -> 100%, std stays at std_end.
      - When keypoint error is below threshold, the coarse reward saturates and stops increasing.

    Reward:
        e_eff = max(||e||, threshold)
        r_i = exp(-e_eff_i^2 / std(t)^2)
    """
    e0, e1, e2 = _ee_kp_errors_lb(
        env=env,
        command_name=command_name,
        asset_cfg=asset_cfg,
        kp_dx=kp_dx,
        kp_dz=kp_dz,
    )

    # phase gate: 0 at cycle start, 1 after gate_end_ratio
    s = _phase_gate_from_command(
        env,
        command_name=command_name,
        start_ratio=gate_start_ratio,
        end_ratio=gate_end_ratio,
        kind=gate_kind,
    )

    # std schedule: std_start -> std_end
    std_t = float(std_start) + s * (float(std_end) - float(std_start))
    std_t = torch.clamp(std_t, min=1e-6)

    # saturate below threshold
    e0_eff = torch.clamp(e0, min=float(threshold))
    e1_eff = torch.clamp(e1, min=float(threshold))
    e2_eff = torch.clamp(e2, min=float(threshold))

    denom = std_t * std_t

    r0 = torch.exp(-(e0_eff * e0_eff) / denom)
    r1 = torch.exp(-(e1_eff * e1_eff) / denom)
    r2 = torch.exp(-(e2_eff * e2_eff) / denom)

    wsum = float(w0 + w1 + w2)
    if wsum < eps:
        return (r0 + r1 + r2) / 3.0

    return (float(w0) * r0 + float(w1) * r1 + float(w2) * r2) / wsum


"""
rewards for loco-manipulation in projected level base frame
"""


# helpers
def _projected_level_base_pose_w(asset, ground_z: float = 0.0):
    """PLB frame in world:
    origin = [base_x, base_y, ground_z]
    orientation = yaw-only(base_quat)
    """
    base_pos_w = asset.data.root_pos_w
    base_quat_w = asset.data.root_quat_w

    plb_pos_w = base_pos_w.clone()
    plb_pos_w[:, 2] = float(ground_z)

    _, _, yaw = euler_xyz_from_quat(base_quat_w)
    zeros = torch.zeros_like(yaw)
    plb_quat_w = quat_from_euler_xyz(zeros, zeros, yaw)

    return plb_pos_w, plb_quat_w


def _ee_kp_errors_plb(
    env,
    command_name: str,
    asset_cfg,
    kp_dx: float = 0.30,
    kp_dz: float = 0.30,
    ground_z: float = 0.0,
):
    """Return e0,e1,e2 keypoint errors in world distance, command is in PLB."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    if command.shape[-1] != 9:
        raise ValueError(
            f"[_ee_kp_errors_plb] Expected command dim 9, got {tuple(command.shape)}"
        )

    kp0_plb = command[:, 0:3]
    kp1_plb = command[:, 3:6]
    kp2_plb = command[:, 6:9]

    plb_pos_w, plb_quat_w = _projected_level_base_pose_w(asset, ground_z=ground_z)

    kp0_des_w, _ = combine_frame_transforms(plb_pos_w, plb_quat_w, kp0_plb)
    kp1_des_w, _ = combine_frame_transforms(plb_pos_w, plb_quat_w, kp1_plb)
    kp2_des_w, _ = combine_frame_transforms(plb_pos_w, plb_quat_w, kp2_plb)

    body_id = asset_cfg.body_ids[0]
    if isinstance(body_id, (list, tuple)):
        body_id = body_id[0]
    body_id = int(body_id)

    ee_pos_w = asset.data.body_pos_w[:, body_id, :]
    ee_quat_w = asset.data.body_quat_w[:, body_id, :]

    off_x = ee_pos_w.new_tensor([kp_dx, 0.0, 0.0]).unsqueeze(0).expand_as(ee_pos_w)
    off_z = ee_pos_w.new_tensor([0.0, 0.0, kp_dz]).unsqueeze(0).expand_as(ee_pos_w)

    kp0_cur_w = ee_pos_w
    kp1_cur_w = ee_pos_w + quat_apply(ee_quat_w, off_x)
    kp2_cur_w = ee_pos_w + quat_apply(ee_quat_w, off_z)

    e0 = torch.linalg.norm(kp0_cur_w - kp0_des_w, dim=-1)
    e1 = torch.linalg.norm(kp1_cur_w - kp1_des_w, dim=-1)
    e2 = torch.linalg.norm(kp2_cur_w - kp2_des_w, dim=-1)

    return e0, e1, e2


# rewards
def ee_kp_tracking_exp_saturated_std_schedule_plb(
    env,
    command_name: str,
    asset_cfg,
    std_start: float = 0.80,
    std_end: float = 0.50,
    kp_dx: float = 0.30,
    kp_dz: float = 0.30,
    ground_z: float = 0.0,
    w0: float = 1.2,
    w1: float = 1.0,
    w2: float = 1.0,
    threshold: float = 0.20,
    gate_start_ratio: float = 0.0,
    gate_end_ratio: float = 0.50,
    gate_kind: str = "smootherstep",
    eps: float = 1e-6,
) -> torch.Tensor:
    e0, e1, e2 = _ee_kp_errors_plb(
        env=env,
        command_name=command_name,
        asset_cfg=asset_cfg,
        kp_dx=kp_dx,
        kp_dz=kp_dz,
        ground_z=ground_z,
    )

    s = _phase_gate_from_command(
        env,
        command_name=command_name,
        start_ratio=gate_start_ratio,
        end_ratio=gate_end_ratio,
        kind=gate_kind,
    )

    std_t = float(std_start) + s * (float(std_end) - float(std_start))
    std_t = torch.clamp(std_t, min=1e-6)

    e0_eff = torch.clamp(e0, min=float(threshold))
    e1_eff = torch.clamp(e1, min=float(threshold))
    e2_eff = torch.clamp(e2, min=float(threshold))

    denom = std_t * std_t

    r0 = torch.exp(-(e0_eff * e0_eff) / denom)
    r1 = torch.exp(-(e1_eff * e1_eff) / denom)
    r2 = torch.exp(-(e2_eff * e2_eff) / denom)

    wsum = float(w0 + w1 + w2)
    if wsum < eps:
        return (r0 + r1 + r2) / 3.0

    return (float(w0) * r0 + float(w1) * r1 + float(w2) * r2) / wsum


def ee_kp_tracking_delayed_exp_plb(
    env,
    command_name: str,
    asset_cfg,
    std: float = 0.10,
    kp_dx: float = 0.30,
    kp_dz: float = 0.30,
    ground_z: float = 0.0,
    w0: float = 1.2,
    w1: float = 1.0,
    w2: float = 1.0,
    std0: float | None = None,
    std1: float | None = None,
    std2: float | None = None,
    gate_start_ratio: float = 0.35,
    gate_end_ratio: float = 0.75,
    gate_kind: str = "smootherstep",
    eps: float = 1e-6,
) -> torch.Tensor:
    e0, e1, e2 = _ee_kp_errors_plb(
        env=env,
        command_name=command_name,
        asset_cfg=asset_cfg,
        kp_dx=kp_dx,
        kp_dz=kp_dz,
        ground_z=ground_z,
    )

    e0_2 = e0 * e0
    e1_2 = e1 * e1
    e2_2 = e2 * e2

    s0 = max(float(std if std0 is None else std0), 1e-6)
    s1 = max(float(std if std1 is None else std1), 1e-6)
    s2 = max(float(std if std2 is None else std2), 1e-6)

    r0 = torch.exp(-e0_2 / (s0 * s0))
    r1 = torch.exp(-e1_2 / (s1 * s1))
    r2 = torch.exp(-e2_2 / (s2 * s2))

    wsum = float(w0 + w1 + w2)
    if wsum < eps:
        r_track = (r0 + r1 + r2) / 3.0
    else:
        r_track = (float(w0) * r0 + float(w1) * r1 + float(w2) * r2) / wsum

    late_gate = _phase_gate_from_command(
        env,
        command_name=command_name,
        start_ratio=gate_start_ratio,
        end_ratio=gate_end_ratio,
        kind=gate_kind,
    )

    return late_gate * r_track


def ee_kp_tracking_sparse_success_plb(
    env,
    command_name: str,
    asset_cfg,
    kp_dx: float = 0.30,
    kp_dz: float = 0.30,
    ground_z: float = 0.0,
    w0: float = 1.2,
    w1: float = 1.0,
    w2: float = 1.0,
    th1: float = 0.10,
    th2: float = 0.05,
    th3: float = 0.03,
    th4: float = 0.01,
    bonus1: float = 0.2,
    bonus2: float = 0.2,
    bonus3: float = 0.3,
    bonus4: float = 0.3,
    metric: str = "weighted_mean",
    eps: float = 1e-6,
) -> torch.Tensor:
    e0, e1, e2 = _ee_kp_errors_plb(
        env=env,
        command_name=command_name,
        asset_cfg=asset_cfg,
        kp_dx=kp_dx,
        kp_dz=kp_dz,
        ground_z=ground_z,
    )

    if metric == "weighted_mean":
        wsum = max(float(w0 + w1 + w2), eps)
        err = (float(w0) * e0 + float(w1) * e1 + float(w2) * e2) / wsum
    elif metric == "max":
        err = torch.maximum(torch.maximum(e0, e1), e2)
    else:
        raise ValueError(f"[ee_kp_tracking_sparse_success_plb] Unsupported metric: {metric}")

    reward = torch.zeros_like(err)
    reward = reward + float(bonus1) * (err < float(th1)).float()
    reward = reward + float(bonus2) * (err < float(th2)).float()
    reward = reward + float(bonus3) * (err < float(th3)).float()
    reward = reward + float(bonus4) * (err < float(th4)).float()
    return reward


def ee_kp_tracking_exp_plb(
    env,
    command_name: str,
    asset_cfg,
    std: float = 0.50,
    kp_dx: float = 0.30,
    kp_dz: float = 0.30,
    ground_z: float = 0.0,
    w0: float = 1.0,
    w1: float = 1.0,
    w2: float = 1.0,
    std0: float | None = None,
    std1: float | None = None,
    std2: float | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """PLB-frame EE keypoint tracking reward with standard exponential kernel.

    Command format:
        [kp0(3), kp1(3), kp2(3)] in PLB frame.

    Reward:
        r_i = exp(-||e_i||^2 / std_i^2)
        r = weighted_average(r0, r1, r2)

    Args:
        std: shared std for all keypoints if std0/std1/std2 are not specified.
        std0/std1/std2: optional per-keypoint std.
        w0/w1/w2: keypoint weights.
            kp0 mainly constrains EE position.
            kp1/kp2 constrain EE orientation through offset keypoints.
    """
    e0, e1, e2 = _ee_kp_errors_plb(
        env=env,
        command_name=command_name,
        asset_cfg=asset_cfg,
        kp_dx=kp_dx,
        kp_dz=kp_dz,
        ground_z=ground_z,
    )

    e0_2 = e0 * e0
    e1_2 = e1 * e1
    e2_2 = e2 * e2

    s0 = max(float(std if std0 is None else std0), eps)
    s1 = max(float(std if std1 is None else std1), eps)
    s2 = max(float(std if std2 is None else std2), eps)

    r0 = torch.exp(-e0_2 / (s0 * s0))
    r1 = torch.exp(-e1_2 / (s1 * s1))
    r2 = torch.exp(-e2_2 / (s2 * s2))

    wsum = float(w0 + w1 + w2)
    if wsum < eps:
        return (r0 + r1 + r2) / 3.0

    return (float(w0) * r0 + float(w1) * r1 + float(w2) * r2) / wsum