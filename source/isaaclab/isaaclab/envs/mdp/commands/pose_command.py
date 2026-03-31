# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

import torch
import os
import numpy as np
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique, euler_xyz_from_quat, quat_apply, quat_inv
from isaaclab.utils.math import _quat_from_keypoints_lb, _quat_slerp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import UniformPoseCommandCfg
    from .commands_cfg import PresampledKeypointsCommandLBCfg
    from .commands_cfg import PresampledKeypointsInterpolateCommandLBCfg


class UniformPoseCommand(CommandTerm):
    """Command generator for generating pose commands uniformly.

    The command generator generates poses by sampling positions uniformly within specified
    regions in cartesian space. For orientation, it samples uniformly the euler angles
    (roll-pitch-yaw) and converts them into quaternion representation (w, x, y, z).

    The position and orientation commands are generated in the base frame of the robot, and not the
    simulation world frame. This means that users need to handle the transformation from the
    base frame to the simulation world frame themselves.

    .. caution::

        Sampling orientations uniformly is not strictly the same as sampling euler angles uniformly.
        This is because rotations are defined by 3D non-Euclidean space, and the mapping
        from euler angles to rotations is not one-to-one.

    """

    cfg: UniformPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformPoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.pose_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # transform command from base frame to simulation world frame
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_pos_w[:, self.body_idx],
            self.robot.data.body_quat_w[:, self.body_idx],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new pose targets
        # -- position
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)
        # -- orientation
        euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        # make sure the quaternion has real part as positive
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        # -- current body pose
        body_link_pose_w = self.robot.data.body_link_pose_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_link_pose_w[:, :3], body_link_pose_w[:, 3:7])


class PresampledKeypointsCommandLB(CommandTerm):
    """Command generator that samples keypoint commands from a precomputed table (Level-Base Frame).

    LB frame:
    - Origin: base (x,y,z) in world
    - Orientation: base yaw kept, roll/pitch = 0

    Table format (N,9): [kp0(3), kp1(3), kp2(3)] in LB.

    Command output:
    - keypoints_command_lb: (num_envs, 9)
    - keypoints_command_w:  (num_envs, 9)  (LB2World transformed)

    Metrics:
    - kp0_error, kp1_error, kp2_error (L2 distance in world)
    - position_error (alias of kp0_error)
    """

    cfg: PresampledKeypointsCommandLBCfg

    def __init__(self, cfg: PresampledKeypointsCommandLBCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # load table (N,9) in LB
        path = cfg.file_path
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)

        arr = np.load(path).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 9:
            raise ValueError(
                f"[PresampledKeypointsCommandLB] Expected npy shape (N,9), got {arr.shape} from '{path}'."
            )

        self._table = torch.from_numpy(arr).to(self.device)  # (N,9)
        self._num_rows = int(self._table.shape[0])

        if hasattr(self.cfg, "sample_mode") and self.cfg.sample_mode != "random":
            raise ValueError("[PresampledKeypointsCommandLB] Only sample_mode='random' is supported.")

        # command buffers
        self.keypoints_command_lb = torch.zeros(self.num_envs, 9, device=self.device)
        self.keypoints_command_w = torch.zeros_like(self.keypoints_command_lb)

        # metrics buffers
        self.metrics["kp0_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["kp1_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["kp2_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)

        # For computing current keypoints in world (must match sampling definition)
        self._dx = float(getattr(cfg, "kp_dx", 0.30))
        self._dz = float(getattr(cfg, "kp_dz", 0.30))

        self._off_x = torch.tensor([self._dx, 0.0, 0.0], device=self.device, dtype=torch.float32)
        self._off_z = torch.tensor([0.0, 0.0, self._dz], device=self.device, dtype=torch.float32)

    def __str__(self) -> str:
        msg = "PresampledKeypointsCommandLB:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tTable rows: {self._num_rows}\n"
        msg += "\tSample mode: random\n"
        msg += "\tFrame: world-level LB (yaw-only, origin at base)\n"
        msg += f"\tKeypoints: kp0=pos, kp1=+X*{self._dx:.3f}, kp2=+Z*{self._dz:.3f}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired keypoints command in LB frame. Shape (num_envs, 9)."""
        return self.keypoints_command_lb

    # helpers
    def _pick_indices(self, k: int) -> torch.Tensor:
        return torch.randint(0, self._num_rows, (k,), device=self.device)

    def _level_base_pose_w(self):
        """Return (pos_w, quat_w) of LB for each env.
        pos_w: (N,3) base origin in world (keep z)
        quat_w: (N,4) yaw-only quaternion (wxyz)
        """
        base_pos_w = self.robot.data.root_pos_w      # (N,3)
        base_quat_w = self.robot.data.root_quat_w    # (N,4) wxyz

        lb_pos_w = base_pos_w.clone()                # keep z (NOT projected)

        roll, pitch, yaw = euler_xyz_from_quat(base_quat_w)
        zeros = torch.zeros_like(yaw)
        lb_quat_w = quat_from_euler_xyz(zeros, zeros, yaw)  # (N,4)

        return lb_pos_w, lb_quat_w

    def _lb_points_to_world(self, lb_pos_w: torch.Tensor, lb_quat_w: torch.Tensor, pts_lb: torch.Tensor) -> torch.Tensor:
        """Transform points from LB to world.
        pts_lb: (N,3)
        returns pts_w: (N,3)
        """
        return lb_pos_w + quat_apply(lb_quat_w, pts_lb)

    def _split_kps(self, kps_9: torch.Tensor):
        kp0 = kps_9[:, 0:3]
        kp1 = kps_9[:, 3:6]
        kp2 = kps_9[:, 6:9]
        return kp0, kp1, kp2

    def _current_keypoints_w(self):
        """Compute current (kp0,kp1,kp2) in world from EE pose in world."""
        ee_pos_w = self.robot.data.body_pos_w[:, self.body_idx, :]     # (N,3)
        ee_quat_w = self.robot.data.body_quat_w[:, self.body_idx, :]   # (N,4) wxyz

        kp0_w = ee_pos_w
        off_x = self._off_x.unsqueeze(0).expand_as(ee_pos_w)
        off_z = self._off_z.unsqueeze(0).expand_as(ee_pos_w)
        kp1_w = ee_pos_w + quat_apply(ee_quat_w, off_x)
        kp2_w = ee_pos_w + quat_apply(ee_quat_w, off_z)
        return kp0_w, kp1_w, kp2_w

    # CommandTerm implementation
    def _resample_command(self, env_ids: Sequence[int]):
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        k = env_ids_t.numel()

        idx = self._pick_indices(k)
        kps = self._table[idx].clone()  # (k,9) in LB
        self.keypoints_command_lb[env_ids_t] = kps

    def _update_command(self):
        # no temporal filtering
        pass

    def _update_metrics(self):
        # LB -> World for each keypoint
        lb_pos_w, lb_quat_w = self._level_base_pose_w()

        kp0_lb, kp1_lb, kp2_lb = self._split_kps(self.keypoints_command_lb)
        kp0_cmd_w = self._lb_points_to_world(lb_pos_w, lb_quat_w, kp0_lb)
        kp1_cmd_w = self._lb_points_to_world(lb_pos_w, lb_quat_w, kp1_lb)
        kp2_cmd_w = self._lb_points_to_world(lb_pos_w, lb_quat_w, kp2_lb)

        # pack world command (useful for debug vis)
        self.keypoints_command_w[:, 0:3] = kp0_cmd_w
        self.keypoints_command_w[:, 3:6] = kp1_cmd_w
        self.keypoints_command_w[:, 6:9] = kp2_cmd_w

        # current keypoints in world
        kp0_cur_w, kp1_cur_w, kp2_cur_w = self._current_keypoints_w()

        # errors
        e0 = kp0_cmd_w - kp0_cur_w
        e1 = kp1_cmd_w - kp1_cur_w
        e2 = kp2_cmd_w - kp2_cur_w

        self.metrics["kp0_error"] = torch.linalg.norm(e0, dim=-1)
        self.metrics["kp1_error"] = torch.linalg.norm(e1, dim=-1)
        self.metrics["kp2_error"] = torch.linalg.norm(e2, dim=-1)
        self.metrics["position_error"].copy_(self.metrics["kp0_error"])

    # Debug visualization
    def _set_debug_vis_impl(self, debug_vis: bool):
        if not debug_vis:
            if hasattr(self, "kp_goal_vis"):
                self.kp_goal_vis.set_visibility(False)
                self.kp_cur_vis.set_visibility(False)
            return

        if not hasattr(self, "kp_goal_vis"):
            self.kp_goal_vis = VisualizationMarkers(self.cfg.goal_kp_visualizer_cfg)
            self.kp_cur_vis = VisualizationMarkers(self.cfg.current_kp_visualizer_cfg)

        self.kp_goal_vis.set_visibility(True)
        self.kp_cur_vis.set_visibility(True)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        # command points in world
        lb_pos_w, lb_quat_w = self._level_base_pose_w()
        kp0_lb, kp1_lb, kp2_lb = self._split_kps(self.keypoints_command_lb)
        kp0_cmd_w = self._lb_points_to_world(lb_pos_w, lb_quat_w, kp0_lb)
        kp1_cmd_w = self._lb_points_to_world(lb_pos_w, lb_quat_w, kp1_lb)
        kp2_cmd_w = self._lb_points_to_world(lb_pos_w, lb_quat_w, kp2_lb)

        goal_pts = torch.cat([kp0_cmd_w, kp1_cmd_w, kp2_cmd_w], dim=0)
        self.kp_goal_vis.visualize(goal_pts)

        # current points in world
        kp0_cur_w, kp1_cur_w, kp2_cur_w = self._current_keypoints_w()
        cur_pts = torch.cat([kp0_cur_w, kp1_cur_w, kp2_cur_w], dim=0)
        self.kp_cur_vis.visualize(cur_pts)


class PresampledKeypointsInterpolateCommandLB(CommandTerm):
    """Presampled keypoints command in Level-Base Frame, with thresholded interpolation on kp0 and rotation.

    Pipeline:
      - sample kps_s = [kp0,kp1,kp2] in LB from table
      - compute alpha_pos from kp0 distance threshold
      - compute alpha_rot from relative rotation threshold
      - alpha = min(alpha_pos, alpha_rot)
      - if both within threshold => alpha==1 -> accept sample
      - else interpolate:
            kp0_new = kp0_prev + alpha * (kp0_s - kp0_prev)
            quat_new = slerp(quat_prev, quat_s, alpha)
            kp1_new,kp2_new rebuilt using dx,dz
    """

    cfg: PresampledKeypointsInterpolateCommandLBCfg

    def __init__(self, cfg: PresampledKeypointsInterpolateCommandLBCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # load table (N,9) in LB
        path = cfg.file_path
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)

        arr = np.load(path).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 9:
            raise ValueError(
                f"[PresampledKeypointsInterpolateCommandLB] Expected npy shape (N,9), got {arr.shape} from '{path}'."
            )

        self._table = torch.from_numpy(arr).to(self.device)  # (N,9)
        self._num_rows = int(self._table.shape[0])

        if hasattr(self.cfg, "sample_mode") and self.cfg.sample_mode != "random":
            raise ValueError("[PresampledKeypointsInterpolateCommandLB] Only sample_mode='random' is supported.")

        # command buffers
        self.keypoints_command_lb = torch.zeros(self.num_envs, 9, device=self.device)
        self.keypoints_command_w = torch.zeros_like(self.keypoints_command_lb)

        # metrics buffers
        self.metrics["kp0_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["kp1_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["kp2_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)

        # sampling definition
        self._dx = float(getattr(cfg, "kp_dx", 0.30))
        self._dz = float(getattr(cfg, "kp_dz", 0.30))
        self._off_x = torch.tensor([self._dx, 0.0, 0.0], device=self.device, dtype=torch.float32)
        self._off_z = torch.tensor([0.0, 0.0, self._dz], device=self.device, dtype=torch.float32)

        # thresholds
        self._kp0_threshold = float(getattr(cfg, "kp0_threshold", 0.20))
        # rotation threshold in radians (e.g. 20deg = 0.349)
        self._rot_threshold = float(getattr(cfg, "rot_threshold", 0.40))

        # per-env init flag: first resample uses raw sample
        self._has_cmd = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def __str__(self) -> str:
        msg = "PresampledKeypointsInterpolateCommandLB:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tTable rows: {self._num_rows}\n"
        msg += "\tSample mode: random\n"
        msg += "\tFrame: world-level LB (yaw-only, origin at base)\n"
        msg += f"\tKeypoints: kp0=pos, kp1=+X*{self._dx:.3f}, kp2=+Z*{self._dz:.3f}\n"
        msg += f"\tInterpolation: kp0_threshold={self._kp0_threshold:.3f}m, rot_threshold={self._rot_threshold:.3f}rad\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        return self.keypoints_command_lb

    # helpers
    def _pick_indices(self, k: int) -> torch.Tensor:
        return torch.randint(0, self._num_rows, (k,), device=self.device)

    def _level_base_pose_w(self) -> Tuple[torch.Tensor, torch.Tensor]:
        base_pos_w = self.robot.data.root_pos_w
        base_quat_w = self.robot.data.root_quat_w
        lb_pos_w = base_pos_w.clone()
        roll, pitch, yaw = euler_xyz_from_quat(base_quat_w)
        zeros = torch.zeros_like(yaw)
        lb_quat_w = quat_from_euler_xyz(zeros, zeros, yaw)
        return lb_pos_w, lb_quat_w

    def _lb_points_to_world(self, lb_pos_w: torch.Tensor, lb_quat_w: torch.Tensor, pts_lb: torch.Tensor) -> torch.Tensor:
        return lb_pos_w + quat_apply(lb_quat_w, pts_lb)

    def _split_kps(self, kps_9: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kp0 = kps_9[:, 0:3]
        kp1 = kps_9[:, 3:6]
        kp2 = kps_9[:, 6:9]
        return kp0, kp1, kp2

    def _pack_kps(self, kp0: torch.Tensor, kp1: torch.Tensor, kp2: torch.Tensor) -> torch.Tensor:
        return torch.cat([kp0, kp1, kp2], dim=-1)

    def _kps_from_pose(self, kp0: torch.Tensor, quat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        off_x = self._off_x.unsqueeze(0).expand_as(kp0)
        off_z = self._off_z.unsqueeze(0).expand_as(kp0)
        kp1 = kp0 + quat_apply(quat, off_x)
        kp2 = kp0 + quat_apply(quat, off_z)
        return kp1, kp2

    def _current_keypoints_w(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ee_pos_w = self.robot.data.body_pos_w[:, self.body_idx, :]
        ee_quat_w = self.robot.data.body_quat_w[:, self.body_idx, :]
        kp0_w = ee_pos_w
        off_x = self._off_x.unsqueeze(0).expand_as(ee_pos_w)
        off_z = self._off_z.unsqueeze(0).expand_as(ee_pos_w)
        kp1_w = ee_pos_w + quat_apply(ee_quat_w, off_x)
        kp2_w = ee_pos_w + quat_apply(ee_quat_w, off_z)
        return kp0_w, kp1_w, kp2_w

    def _quat_angle(self, q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
        """Return relative rotation angle (rad) between two quaternions (wxyz)."""
        # shortest path via abs(dot)
        dot = torch.abs(torch.sum(q0 * q1, dim=-1)).clamp(0.0, 1.0)
        return 2.0 * torch.acos(dot)

    # CommandTerm implementation
    def _resample_command(self, env_ids: Sequence[int]):
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        k = env_ids_t.numel()
        if k == 0:
            return

        # 1) sample candidates from table
        idx = self._pick_indices(k)
        kps_s = self._table[idx].clone()  # (k,9) in LB
        kp0_s, kp1_s, kp2_s = self._split_kps(kps_s)

        # 2) build "previous reference"
        #    - if has previous command: use previous command
        #    - else: use current EE keypoints (converted to LB)
        has_p = self._has_cmd[env_ids_t]  # (k,)

        # previous command in LB
        kps_prev_cmd = self.keypoints_command_lb[env_ids_t].clone()

        # current EE keypoints in world
        kp0_cur_w, kp1_cur_w, kp2_cur_w = self._current_keypoints_w()
        kp0_cur_w = kp0_cur_w[env_ids_t]
        kp1_cur_w = kp1_cur_w[env_ids_t]
        kp2_cur_w = kp2_cur_w[env_ids_t]

        # world -> LB
        lb_pos_w, lb_quat_w = self._level_base_pose_w()
        lb_pos_w = lb_pos_w[env_ids_t]
        lb_quat_w = lb_quat_w[env_ids_t]

        # translate into LB origin
        kp0_cur_rel_w = kp0_cur_w - lb_pos_w
        kp1_cur_rel_w = kp1_cur_w - lb_pos_w
        kp2_cur_rel_w = kp2_cur_w - lb_pos_w

        # rotate world -> LB  => apply inverse yaw quat
        lb_quat_inv_w = quat_inv(lb_quat_w)
        kp0_cur_lb = quat_apply(lb_quat_inv_w, kp0_cur_rel_w)
        kp1_cur_lb = quat_apply(lb_quat_inv_w, kp1_cur_rel_w)
        kp2_cur_lb = quat_apply(lb_quat_inv_w, kp2_cur_rel_w)

        kps_cur_lb = self._pack_kps(kp0_cur_lb, kp1_cur_lb, kp2_cur_lb)

        # choose previous reference
        kps_p = torch.where(has_p.unsqueeze(-1), kps_prev_cmd, kps_cur_lb)
        kp0_p, kp1_p, kp2_p = self._split_kps(kps_p)

        # 3) reconstruct pose from keypoints
        quat_p = _quat_from_keypoints_lb(kp0_p, kp1_p, kp2_p, self._dx, self._dz)
        quat_s = _quat_from_keypoints_lb(kp0_s, kp1_s, kp2_s, self._dx, self._dz)

        # 4) alpha_pos from kp0 threshold
        delta = kp0_s - kp0_p
        dist = torch.linalg.norm(delta, dim=-1).clamp_min(1e-8)  # (k,)
        alpha_pos = (self._kp0_threshold / dist).clamp(max=1.0)

        # 5) alpha_rot from rotation threshold
        ang = self._quat_angle(quat_p, quat_s).clamp_min(1e-8)   # (k,)
        alpha_rot = (self._rot_threshold / ang).clamp(max=1.0)

        # 6) final alpha
        alpha = torch.minimum(alpha_pos, alpha_rot)  # (k,)

        # 7) if already within both thresholds, accept raw sample directly
        within = (dist <= self._kp0_threshold) & (ang <= self._rot_threshold)
        alpha_eff = torch.where(within, torch.ones_like(alpha), alpha)

        # 8) interpolate
        kp0_new = kp0_p + alpha_eff.unsqueeze(-1) * delta
        quat_new = _quat_slerp(quat_p, quat_s, alpha_eff)
        kp1_new, kp2_new = self._kps_from_pose(kp0_new, quat_new)
        kps_interp = self._pack_kps(kp0_new, kp1_new, kp2_new)

        # 9) save
        self.keypoints_command_lb[env_ids_t] = kps_interp
        self._has_cmd[env_ids_t] = True

    def _update_command(self):
        pass

    def _update_metrics(self):
        lb_pos_w, lb_quat_w = self._level_base_pose_w()

        kp0_lb, kp1_lb, kp2_lb = self._split_kps(self.keypoints_command_lb)
        kp0_cmd_w = self._lb_points_to_world(lb_pos_w, lb_quat_w, kp0_lb)
        kp1_cmd_w = self._lb_points_to_world(lb_pos_w, lb_quat_w, kp1_lb)
        kp2_cmd_w = self._lb_points_to_world(lb_pos_w, lb_quat_w, kp2_lb)

        self.keypoints_command_w[:, 0:3] = kp0_cmd_w
        self.keypoints_command_w[:, 3:6] = kp1_cmd_w
        self.keypoints_command_w[:, 6:9] = kp2_cmd_w

        kp0_cur_w, kp1_cur_w, kp2_cur_w = self._current_keypoints_w()

        self.metrics["kp0_error"] = torch.linalg.norm(kp0_cmd_w - kp0_cur_w, dim=-1)
        self.metrics["kp1_error"] = torch.linalg.norm(kp1_cmd_w - kp1_cur_w, dim=-1)
        self.metrics["kp2_error"] = torch.linalg.norm(kp2_cmd_w - kp2_cur_w, dim=-1)
        self.metrics["position_error"].copy_(self.metrics["kp0_error"])

    def _set_debug_vis_impl(self, debug_vis: bool):
        if not debug_vis:
            if hasattr(self, "kp_goal_vis"):
                self.kp_goal_vis.set_visibility(False)
                self.kp_cur_vis.set_visibility(False)
            return

        if not hasattr(self, "kp_goal_vis"):
            self.kp_goal_vis = VisualizationMarkers(self.cfg.goal_kp_visualizer_cfg)
            self.kp_cur_vis = VisualizationMarkers(self.cfg.current_kp_visualizer_cfg)

        self.kp_goal_vis.set_visibility(True)
        self.kp_cur_vis.set_visibility(True)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        lb_pos_w, lb_quat_w = self._level_base_pose_w()
        kp0_lb, kp1_lb, kp2_lb = self._split_kps(self.keypoints_command_lb)
        kp0_cmd_w = self._lb_points_to_world(lb_pos_w, lb_quat_w, kp0_lb)
        kp1_cmd_w = self._lb_points_to_world(lb_pos_w, lb_quat_w, kp1_lb)
        kp2_cmd_w = self._lb_points_to_world(lb_pos_w, lb_quat_w, kp2_lb)

        goal_pts = torch.cat([kp0_cmd_w, kp1_cmd_w, kp2_cmd_w], dim=0)
        self.kp_goal_vis.visualize(goal_pts)

        kp0_cur_w, kp1_cur_w, kp2_cur_w = self._current_keypoints_w()
        cur_pts = torch.cat([kp0_cur_w, kp1_cur_w, kp2_cur_w], dim=0)
        self.kp_cur_vis.visualize(cur_pts)