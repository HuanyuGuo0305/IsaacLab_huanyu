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

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import (
    combine_frame_transforms, 
    compute_pose_error, 
    quat_from_euler_xyz, 
    quat_unique, 
    euler_xyz_from_quat, 
    quat_apply, 
    quat_inv,
    _quat_from_keypoints_lb,
    _quat_slerp,)

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import UniformPoseCommandCfg
    from .commands_cfg import PresampledKeypointsDirectCommandLBCfg
    from .commands_cfg import PresampledKeypointsCubicTrajectoryCommandLBCfg
    from .commands_cfg import PresampledKeypointsDirectCommandPLBCfg
    from .commands_cfg import PresampledKeypointsCubicTrajectoryCommandPLBCfg


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



class PresampledKeypointsDirectCommandLB(CommandTerm):
    """Directly sampled EE keypoint command in LB frame.

    Behavior:
      1) Randomly sample one row from a presampled reachable LB keypoint table
      2) Hold this sampled keypoint command until next resampling
      3) No cubic interpolation
      4) No adjacent-target kp0 threshold clipping

    Command format:
      [kp0(3), kp1(3), kp2(3)] in LB frame.
    """

    cfg: "PresampledKeypointsDirectCommandLBCfg"

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        path = cfg.file_path
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)

        arr = np.load(path).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 9:
            raise ValueError(
                f"[PresampledKeypointsDirectCommandLB] "
                f"Expected npy shape (N, 9), got {arr.shape} from '{path}'."
            )

        self._table = torch.from_numpy(arr).to(self.device)
        self._num_rows = int(self._table.shape[0])

        if getattr(self.cfg, "sample_mode", "random") != "random":
            raise ValueError(
                "[PresampledKeypointsDirectCommandLB] "
                "Only sample_mode='random' is supported."
            )

        self._dx = float(getattr(cfg, "kp_dx", 0.30))
        self._dz = float(getattr(cfg, "kp_dz", 0.30))
        self._off_x = torch.tensor([self._dx, 0.0, 0.0], device=self.device, dtype=torch.float32)
        self._off_z = torch.tensor([0.0, 0.0, self._dz], device=self.device, dtype=torch.float32)

        # Current command in LB / world
        self.keypoints_command_lb = torch.zeros(self.num_envs, 9, device=self.device)
        self.keypoints_command_w = torch.zeros_like(self.keypoints_command_lb)

        # Whether a valid command has been assigned
        self._has_cmd = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Metrics
        self.metrics["kp0_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["kp1_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["kp2_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "PresampledKeypointsDirectCommandLB:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tTable rows: {self._num_rows}\n"
        msg += "\tSample mode: random\n"
        msg += "\tFrame: LB (level-base yaw-only frame)\n"
        msg += f"\tKeypoints: kp0=pos, kp1=+X*{self._dx:.3f}, kp2=+Z*{self._dz:.3f}\n"
        msg += "\tMode: direct sample-and-hold from presampled reachable set\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        return self.keypoints_command_lb

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)

        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # Clear state
        self._has_cmd[env_ids_t] = False
        self.keypoints_command_lb[env_ids_t] = 0.0
        self.keypoints_command_w[env_ids_t] = 0.0

        # Sample a fresh command immediately after reset
        self._resample_command(env_ids_t)

        return extras

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

    def _lb_points_to_world(
        self, lb_pos_w: torch.Tensor, lb_quat_w: torch.Tensor, pts_lb: torch.Tensor
    ) -> torch.Tensor:
        return lb_pos_w + quat_apply(lb_quat_w, pts_lb)

    def _split_kps(self, kps_9: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kp0 = kps_9[:, 0:3]
        kp1 = kps_9[:, 3:6]
        kp2 = kps_9[:, 6:9]
        return kp0, kp1, kp2

    def _pack_kps(self, kp0: torch.Tensor, kp1: torch.Tensor, kp2: torch.Tensor) -> torch.Tensor:
        return torch.cat([kp0, kp1, kp2], dim=-1)

    def _current_keypoints_w(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ee_pos_w = self.robot.data.body_pos_w[:, self.body_idx, :]
        ee_quat_w = self.robot.data.body_quat_w[:, self.body_idx, :]

        kp0_w = ee_pos_w
        off_x = self._off_x.unsqueeze(0).expand_as(ee_pos_w)
        off_z = self._off_z.unsqueeze(0).expand_as(ee_pos_w)
        kp1_w = ee_pos_w + quat_apply(ee_quat_w, off_x)
        kp2_w = ee_pos_w + quat_apply(ee_quat_w, off_z)
        return kp0_w, kp1_w, kp2_w

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        k = env_ids_t.numel()
        if k == 0:
            return

        idx = self._pick_indices(k)
        sampled = self._table[idx].clone()

        self.keypoints_command_lb[env_ids_t] = sampled
        self._has_cmd[env_ids_t] = True

    def _update_command(self):
        # Direct command: hold sampled target until next resample.
        # Nothing to do here.
        pass

    def _update_metrics(self):
        lb_pos_w, lb_quat_w = self._level_base_pose_w()

        # Current command in world
        kp0_lb, kp1_lb, kp2_lb = self._split_kps(self.keypoints_command_lb)
        kp0_cmd_w = self._lb_points_to_world(lb_pos_w, lb_quat_w, kp0_lb)
        kp1_cmd_w = self._lb_points_to_world(lb_pos_w, lb_quat_w, kp1_lb)
        kp2_cmd_w = self._lb_points_to_world(lb_pos_w, lb_quat_w, kp2_lb)

        self.keypoints_command_w[:, 0:3] = kp0_cmd_w
        self.keypoints_command_w[:, 3:6] = kp1_cmd_w
        self.keypoints_command_w[:, 6:9] = kp2_cmd_w

        # Current EE
        kp0_cur_w, kp1_cur_w, kp2_cur_w = self._current_keypoints_w()

        # Tracking metrics
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

        # Visualize sampled command
        lb_pos_w, lb_quat_w = self._level_base_pose_w()
        cmd_kp0_lb, cmd_kp1_lb, cmd_kp2_lb = self._split_kps(self.keypoints_command_lb)

        cmd_kp0_w = self._lb_points_to_world(lb_pos_w, lb_quat_w, cmd_kp0_lb)
        cmd_kp1_w = self._lb_points_to_world(lb_pos_w, lb_quat_w, cmd_kp1_lb)
        cmd_kp2_w = self._lb_points_to_world(lb_pos_w, lb_quat_w, cmd_kp2_lb)

        goal_pts = torch.cat([cmd_kp0_w, cmd_kp1_w, cmd_kp2_w], dim=0)
        self.kp_goal_vis.visualize(goal_pts)

        # Visualize current EE
        kp0_cur_w, kp1_cur_w, kp2_cur_w = self._current_keypoints_w()
        cur_pts = torch.cat([kp0_cur_w, kp1_cur_w, kp2_cur_w], dim=0)
        self.kp_cur_vis.visualize(cur_pts)


class PresampledKeypointsDirectCommandPLB(CommandTerm):
    """Directly sampled EE keypoint command in PLB frame.

    PLB frame:
      - origin = [base_x, base_y, ground_z]
      - orientation = yaw-only(base_quat)

    Command format:
      [kp0(3), kp1(3), kp2(3)] in PLB frame.
    """

    cfg: "PresampledKeypointsDirectCommandPLBCfg"

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        path = cfg.file_path
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)

        arr = np.load(path).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 9:
            raise ValueError(
                f"[PresampledKeypointsDirectCommandPLB] "
                f"Expected npy shape (N, 9), got {arr.shape} from '{path}'."
            )

        self._table = torch.from_numpy(arr).to(self.device)
        self._num_rows = int(self._table.shape[0])

        if getattr(self.cfg, "sample_mode", "random") != "random":
            raise ValueError(
                "[PresampledKeypointsDirectCommandPLB] "
                "Only sample_mode='random' is supported."
            )

        self._dx = float(getattr(cfg, "kp_dx", 0.30))
        self._dz = float(getattr(cfg, "kp_dz", 0.30))
        self._ground_z = float(getattr(cfg, "ground_z", 0.0))

        self._off_x = torch.tensor([self._dx, 0.0, 0.0], device=self.device, dtype=torch.float32)
        self._off_z = torch.tensor([0.0, 0.0, self._dz], device=self.device, dtype=torch.float32)

        self.keypoints_command_plb = torch.zeros(self.num_envs, 9, device=self.device)
        self.keypoints_command_w = torch.zeros_like(self.keypoints_command_plb)

        self._has_cmd = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.metrics["kp0_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["kp1_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["kp2_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "PresampledKeypointsDirectCommandPLB:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tTable rows: {self._num_rows}\n"
        msg += "\tSample mode: random\n"
        msg += "\tFrame: PLB projected level-base yaw-only frame\n"
        msg += f"\tground_z: {self._ground_z:.3f}\n"
        msg += f"\tKeypoints: kp0=pos, kp1=+X*{self._dx:.3f}, kp2=+Z*{self._dz:.3f}\n"
        msg += "\tMode: direct sample-and-hold from presampled reachable set\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        return self.keypoints_command_plb

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)

        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        self._has_cmd[env_ids_t] = False
        self.keypoints_command_plb[env_ids_t] = 0.0
        self.keypoints_command_w[env_ids_t] = 0.0

        self._resample_command(env_ids_t)

        return extras

    def _pick_indices(self, k: int) -> torch.Tensor:
        return torch.randint(0, self._num_rows, (k,), device=self.device)

    def _projected_level_base_pose_w(self) -> Tuple[torch.Tensor, torch.Tensor]:
        base_pos_w = self.robot.data.root_pos_w
        base_quat_w = self.robot.data.root_quat_w

        plb_pos_w = base_pos_w.clone()
        plb_pos_w[:, 2] = self._ground_z

        roll, pitch, yaw = euler_xyz_from_quat(base_quat_w)
        zeros = torch.zeros_like(yaw)
        plb_quat_w = quat_from_euler_xyz(zeros, zeros, yaw)

        return plb_pos_w, plb_quat_w

    def _plb_points_to_world(
        self,
        plb_pos_w: torch.Tensor,
        plb_quat_w: torch.Tensor,
        pts_plb: torch.Tensor,
    ) -> torch.Tensor:
        return plb_pos_w + quat_apply(plb_quat_w, pts_plb)

    def _split_kps(self, kps_9: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kp0 = kps_9[:, 0:3]
        kp1 = kps_9[:, 3:6]
        kp2 = kps_9[:, 6:9]
        return kp0, kp1, kp2

    def _pack_kps(self, kp0: torch.Tensor, kp1: torch.Tensor, kp2: torch.Tensor) -> torch.Tensor:
        return torch.cat([kp0, kp1, kp2], dim=-1)

    def _current_keypoints_w(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ee_pos_w = self.robot.data.body_pos_w[:, self.body_idx, :]
        ee_quat_w = self.robot.data.body_quat_w[:, self.body_idx, :]

        kp0_w = ee_pos_w
        off_x = self._off_x.unsqueeze(0).expand_as(ee_pos_w)
        off_z = self._off_z.unsqueeze(0).expand_as(ee_pos_w)

        kp1_w = ee_pos_w + quat_apply(ee_quat_w, off_x)
        kp2_w = ee_pos_w + quat_apply(ee_quat_w, off_z)

        return kp0_w, kp1_w, kp2_w

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        k = env_ids_t.numel()
        if k == 0:
            return

        idx = self._pick_indices(k)
        sampled = self._table[idx].clone()

        self.keypoints_command_plb[env_ids_t] = sampled
        self._has_cmd[env_ids_t] = True

    def _update_command(self):
        pass

    def _update_metrics(self):
        plb_pos_w, plb_quat_w = self._projected_level_base_pose_w()

        kp0_plb, kp1_plb, kp2_plb = self._split_kps(self.keypoints_command_plb)

        kp0_cmd_w = self._plb_points_to_world(plb_pos_w, plb_quat_w, kp0_plb)
        kp1_cmd_w = self._plb_points_to_world(plb_pos_w, plb_quat_w, kp1_plb)
        kp2_cmd_w = self._plb_points_to_world(plb_pos_w, plb_quat_w, kp2_plb)

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

        plb_pos_w, plb_quat_w = self._projected_level_base_pose_w()

        cmd_kp0_plb, cmd_kp1_plb, cmd_kp2_plb = self._split_kps(self.keypoints_command_plb)

        cmd_kp0_w = self._plb_points_to_world(plb_pos_w, plb_quat_w, cmd_kp0_plb)
        cmd_kp1_w = self._plb_points_to_world(plb_pos_w, plb_quat_w, cmd_kp1_plb)
        cmd_kp2_w = self._plb_points_to_world(plb_pos_w, plb_quat_w, cmd_kp2_plb)

        goal_pts = torch.cat([cmd_kp0_w, cmd_kp1_w, cmd_kp2_w], dim=0)
        self.kp_goal_vis.visualize(goal_pts)

        kp0_cur_w, kp1_cur_w, kp2_cur_w = self._current_keypoints_w()
        cur_pts = torch.cat([kp0_cur_w, kp1_cur_w, kp2_cur_w], dim=0)
        self.kp_cur_vis.visualize(cur_pts)


class PresampledKeypointsCubicTrajectoryCommandLB(CommandTerm):
    """Presampled EE keypoints goal-tracking command in LB frame.

    Behavior:
      1) Sample a fixed goal from a presampled reachable table
      2) Sample a random adjacent-target position threshold
      3) Apply adjacent-target acceptance in position and SLERP orientation by the same alpha
      4) Map accepted goal distance to trajectory duration
      5) Generate a cubic reference from current start pose to the fixed goal
      6) Hold at the fixed goal for the remaining cycle duration

    Command format:
      [kp0(3), kp1(3), kp2(3)] in LB frame.
    """

    cfg: "PresampledKeypointsCubicTrajectoryCommandLBCfg"

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        path = cfg.file_path
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)

        arr = np.load(path).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 9:
            raise ValueError(
                f"[PresampledKeypointsCubicTrajectoryCommandLB] "
                f"Expected npy shape (N,9), got {arr.shape} from '{path}'."
            )

        self._table = torch.from_numpy(arr).to(self.device)
        self._num_rows = int(self._table.shape[0])

        if getattr(self.cfg, "sample_mode", "random") != "random":
            raise ValueError(
                "[PresampledKeypointsCubicTrajectoryCommandLB] "
                "Only sample_mode='random' is supported."
            )

        self._dx = float(getattr(cfg, "kp_dx", 0.30))
        self._dz = float(getattr(cfg, "kp_dz", 0.30))
        self._off_x = torch.tensor([self._dx, 0.0, 0.0], device=self.device, dtype=torch.float32)
        self._off_z = torch.tensor([0.0, 0.0, self._dz], device=self.device, dtype=torch.float32)

        if not (hasattr(cfg, "kp0_threshold_range") and cfg.kp0_threshold_range is not None):
            raise ValueError(
                "[PresampledKeypointsCubicTrajectoryCommandLB] "
                "kp0_threshold_range must be provided."
            )
        self._kp0_threshold_min = float(cfg.kp0_threshold_range[0])
        self._kp0_threshold_max = float(cfg.kp0_threshold_range[1])
        if self._kp0_threshold_min <= 0.0 or self._kp0_threshold_max < self._kp0_threshold_min:
            raise ValueError(
                f"Invalid kp0_threshold_range={cfg.kp0_threshold_range}"
            )

        self._cycle_duration_s = float(getattr(cfg, "cycle_duration_s", 8.0))
        if self._cycle_duration_s <= 0.0:
            raise ValueError(f"Invalid cycle_duration_s={self._cycle_duration_s}")

        self._traj_duration_min_s = float(getattr(cfg, "traj_duration_min_s", 4.0))
        self._traj_duration_max_s = float(getattr(cfg, "traj_duration_max_s", 6.0))
        if self._traj_duration_min_s <= 0.0 or self._traj_duration_max_s < self._traj_duration_min_s:
            raise ValueError(
                f"Invalid traj duration range=({self._traj_duration_min_s}, {self._traj_duration_max_s})"
            )
        if self._traj_duration_max_s > self._cycle_duration_s:
            raise ValueError(
                f"traj_duration_max_s={self._traj_duration_max_s} exceeds cycle_duration_s={self._cycle_duration_s}"
            )

        if hasattr(cfg, "resampling_time_range"):
            low, high = cfg.resampling_time_range
            if abs(float(low) - self._cycle_duration_s) > 1e-6 or abs(float(high) - self._cycle_duration_s) > 1e-6:
                raise ValueError(
                    "[PresampledKeypointsCubicTrajectoryCommandLB] "
                    f"Expected resampling_time_range == ({self._cycle_duration_s}, {self._cycle_duration_s}), "
                    f"got {cfg.resampling_time_range}."
                )

        # Current reference command
        self.keypoints_command_lb = torch.zeros(self.num_envs, 9, device=self.device)
        self.keypoints_command_w = torch.zeros_like(self.keypoints_command_lb)

        # Fixed sampled goal for current cycle
        self.goal_keypoints_lb = torch.zeros(self.num_envs, 9, device=self.device)
        self.goal_keypoints_w = torch.zeros_like(self.goal_keypoints_lb)

        self._has_cmd = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Trajectory start
        self._traj_start_pos_lb = torch.zeros(self.num_envs, 3, device=self.device)
        self._traj_start_quat_lb = torch.zeros(self.num_envs, 4, device=self.device)
        self._traj_start_quat_lb[:, 0] = 1.0

        # Fixed goal pose for current cycle
        self._goal_pos_lb = torch.zeros(self.num_envs, 3, device=self.device)
        self._goal_quat_lb = torch.zeros(self.num_envs, 4, device=self.device)
        self._goal_quat_lb[:, 0] = 1.0

        # Per-env sampled params
        self._kp0_threshold_env = torch.full(
            (self.num_envs,), self._kp0_threshold_min, device=self.device, dtype=torch.float32
        )
        self._traj_duration_env = torch.full(
            (self.num_envs,), self._traj_duration_min_s, device=self.device, dtype=torch.float32
        )
        self._hold_duration_env = torch.full(
            (self.num_envs,),
            self._cycle_duration_s - self._traj_duration_min_s,
            device=self.device,
            dtype=torch.float32,
        )

        # Metrics
        self.metrics["kp0_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["kp1_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["kp2_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)

        self.metrics["goal_kp0_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_kp1_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_kp2_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_position_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "PresampledKeypointsCubicTrajectoryCommandLB:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tTable rows: {self._num_rows}\n"
        msg += "\tSample mode: random\n"
        msg += "\tFrame: LB (level-base yaw-only frame)\n"
        msg += f"\tKeypoints: kp0=pos, kp1=+X*{self._dx:.3f}, kp2=+Z*{self._dz:.3f}\n"
        msg += (
            f"\tAdjacent-target limit: "
            f"kp0_threshold_range=({self._kp0_threshold_min:.3f}, {self._kp0_threshold_max:.3f}) m\n"
        )
        msg += (
            f"\tTiming: cycle={self._cycle_duration_s:.3f}s, "
            f"traj_range=({self._traj_duration_min_s:.3f}, {self._traj_duration_max_s:.3f}) s, "
            f"hold=cycle-traj\n"
        )
        msg += "\tMode: fixed-goal tracking with online cubic reference generation\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        return self.keypoints_command_lb

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)

        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        self._has_cmd[env_ids_t] = False

        self._traj_start_pos_lb[env_ids_t] = 0.0
        self._traj_start_quat_lb[env_ids_t] = 0.0
        self._traj_start_quat_lb[env_ids_t, 0] = 1.0

        self._goal_pos_lb[env_ids_t] = 0.0
        self._goal_quat_lb[env_ids_t] = 0.0
        self._goal_quat_lb[env_ids_t, 0] = 1.0

        self.keypoints_command_lb[env_ids_t] = 0.0
        self.keypoints_command_w[env_ids_t] = 0.0
        self.goal_keypoints_lb[env_ids_t] = 0.0
        self.goal_keypoints_w[env_ids_t] = 0.0

        self._kp0_threshold_env[env_ids_t] = self._kp0_threshold_min
        self._traj_duration_env[env_ids_t] = self._traj_duration_min_s
        self._hold_duration_env[env_ids_t] = self._cycle_duration_s - self._traj_duration_min_s

        return extras

    def _pick_indices(self, k: int) -> torch.Tensor:
        return torch.randint(0, self._num_rows, (k,), device=self.device)

    def _sample_kp0_threshold(self, env_ids_t: torch.Tensor) -> torch.Tensor:
        k = env_ids_t.numel()
        thr = torch.empty(k, device=self.device, dtype=torch.float32).uniform_(
            self._kp0_threshold_min, self._kp0_threshold_max
        )
        self._kp0_threshold_env[env_ids_t] = thr
        return thr

    def _level_base_pose_w(self) -> Tuple[torch.Tensor, torch.Tensor]:
        base_pos_w = self.robot.data.root_pos_w
        base_quat_w = self.robot.data.root_quat_w
        lb_pos_w = base_pos_w.clone()

        roll, pitch, yaw = euler_xyz_from_quat(base_quat_w)
        zeros = torch.zeros_like(yaw)
        lb_quat_w = quat_from_euler_xyz(zeros, zeros, yaw)
        return lb_pos_w, lb_quat_w

    def _lb_points_to_world(
        self, lb_pos_w: torch.Tensor, lb_quat_w: torch.Tensor, pts_lb: torch.Tensor
    ) -> torch.Tensor:
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

    def _current_keypoints_lb(self, env_ids_t: torch.Tensor) -> torch.Tensor:
        kp0_cur_w, kp1_cur_w, kp2_cur_w = self._current_keypoints_w()
        kp0_cur_w = kp0_cur_w[env_ids_t]
        kp1_cur_w = kp1_cur_w[env_ids_t]
        kp2_cur_w = kp2_cur_w[env_ids_t]

        lb_pos_w, lb_quat_w = self._level_base_pose_w()
        lb_pos_w = lb_pos_w[env_ids_t]
        lb_quat_w = lb_quat_w[env_ids_t]
        lb_quat_inv_w = quat_inv(lb_quat_w)

        kp0_cur_lb = quat_apply(lb_quat_inv_w, kp0_cur_w - lb_pos_w)
        kp1_cur_lb = quat_apply(lb_quat_inv_w, kp1_cur_w - lb_pos_w)
        kp2_cur_lb = quat_apply(lb_quat_inv_w, kp2_cur_w - lb_pos_w)

        return self._pack_kps(kp0_cur_lb, kp1_cur_lb, kp2_cur_lb)

    def _cubic_time_scaling(self, tau: torch.Tensor) -> torch.Tensor:
        tau = tau.clamp(0.0, 1.0)
        return 3.0 * tau * tau - 2.0 * tau * tau * tau

    def _apply_adjacent_target_limit(
        self,
        kp0_ref: torch.Tensor,
        quat_ref: torch.Tensor,
        kp0_raw: torch.Tensor,
        quat_raw: torch.Tensor,
        kp0_threshold: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        delta = kp0_raw - kp0_ref
        dist = torch.linalg.norm(delta, dim=-1).clamp_min(1e-8)

        alpha_pos = (kp0_threshold / dist).clamp(max=1.0)
        within = dist <= kp0_threshold
        alpha_eff = torch.where(within, torch.ones_like(alpha_pos), alpha_pos)

        kp0_new = kp0_ref + alpha_eff.unsqueeze(-1) * delta
        quat_new = _quat_slerp(quat_ref, quat_raw, alpha_eff)
        return kp0_new, quat_new

    def _compute_traj_duration_from_distance(
        self,
        env_ids_t: torch.Tensor,
        kp0_start: torch.Tensor,
        kp0_goal: torch.Tensor,
    ) -> torch.Tensor:
        dist_eff = torch.linalg.norm(kp0_goal - kp0_start, dim=-1)

        dist_min = self._kp0_threshold_min
        dist_max = self._kp0_threshold_max

        if dist_max <= dist_min + 1e-8:
            alpha = torch.zeros_like(dist_eff)
        else:
            alpha = ((dist_eff - dist_min) / (dist_max - dist_min)).clamp(0.0, 1.0)

        traj = self._traj_duration_min_s + alpha * (self._traj_duration_max_s - self._traj_duration_min_s)
        hold = self._cycle_duration_s - traj

        self._traj_duration_env[env_ids_t] = traj
        self._hold_duration_env[env_ids_t] = hold

        return traj

    def _eval_current_traj_command(self, env_ids_t: torch.Tensor) -> torch.Tensor:
        t_left = self.time_left[env_ids_t]
        t = (self._cycle_duration_s - t_left).clamp(min=0.0, max=self._cycle_duration_s)

        traj_duration = self._traj_duration_env[env_ids_t].clamp_min(1e-6)
        tau = (t / traj_duration).clamp(0.0, 1.0)
        s = self._cubic_time_scaling(tau)

        start_pos = self._traj_start_pos_lb[env_ids_t]
        start_quat = self._traj_start_quat_lb[env_ids_t]
        goal_pos = self._goal_pos_lb[env_ids_t]
        goal_quat = self._goal_quat_lb[env_ids_t]

        pos = start_pos + s.unsqueeze(-1) * (goal_pos - start_pos)
        quat = _quat_slerp(start_quat, goal_quat, s)

        kp1, kp2 = self._kps_from_pose(pos, quat)
        return self._pack_kps(pos, kp1, kp2)

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        k = env_ids_t.numel()
        if k == 0:
            return

        has_prev = self._has_cmd[env_ids_t]
        kps_prev_cmd = self.keypoints_command_lb[env_ids_t].clone()
        kps_cur_lb = self._current_keypoints_lb(env_ids_t)

        # Start pose for the new cycle
        # If a previous command exists, continue smoothly from the last command.
        # Otherwise start from the current measured EE pose.
        kps_start = torch.where(has_prev.unsqueeze(-1), kps_prev_cmd, kps_cur_lb)
        kp0_start, kp1_start, kp2_start = self._split_kps(kps_start)
        quat_start = _quat_from_keypoints_lb(kp0_start, kp1_start, kp2_start, self._dx, self._dz)

        # Sample a fixed goal for this cycle
        idx = self._pick_indices(k)
        kps_raw = self._table[idx].clone()
        kp0_raw, kp1_raw, kp2_raw = self._split_kps(kps_raw)
        quat_raw = _quat_from_keypoints_lb(kp0_raw, kp1_raw, kp2_raw, self._dx, self._dz)

        kp0_threshold = self._sample_kp0_threshold(env_ids_t)

        kp0_goal, quat_goal = self._apply_adjacent_target_limit(
            kp0_ref=kp0_start,
            quat_ref=quat_start,
            kp0_raw=kp0_raw,
            quat_raw=quat_raw,
            kp0_threshold=kp0_threshold,
        )

        self._compute_traj_duration_from_distance(
            env_ids_t=env_ids_t,
            kp0_start=kp0_start,
            kp0_goal=kp0_goal,
        )

        self._traj_start_pos_lb[env_ids_t] = kp0_start
        self._traj_start_quat_lb[env_ids_t] = quat_start

        self._goal_pos_lb[env_ids_t] = kp0_goal
        self._goal_quat_lb[env_ids_t] = quat_goal

        self._has_cmd[env_ids_t] = True

        # Save the fixed goal keypoints for metrics/debug
        kp1_goal, kp2_goal = self._kps_from_pose(kp0_goal, quat_goal)
        self.goal_keypoints_lb[env_ids_t] = self._pack_kps(kp0_goal, kp1_goal, kp2_goal)

        # Initialize current reference at start of the new cycle
        kp1_start_rebuild, kp2_start_rebuild = self._kps_from_pose(kp0_start, quat_start)
        self.keypoints_command_lb[env_ids_t] = self._pack_kps(kp0_start, kp1_start_rebuild, kp2_start_rebuild)

    def _update_command(self):
        valid_env_ids = torch.nonzero(self._has_cmd, as_tuple=False).squeeze(-1)
        if valid_env_ids.numel() == 0:
            return
        self.keypoints_command_lb[valid_env_ids] = self._eval_current_traj_command(valid_env_ids)

    def _update_metrics(self):
        lb_pos_w, lb_quat_w = self._level_base_pose_w()

        # Current reference in world
        kp0_lb, kp1_lb, kp2_lb = self._split_kps(self.keypoints_command_lb)
        kp0_cmd_w = self._lb_points_to_world(lb_pos_w, lb_quat_w, kp0_lb)
        kp1_cmd_w = self._lb_points_to_world(lb_pos_w, lb_quat_w, kp1_lb)
        kp2_cmd_w = self._lb_points_to_world(lb_pos_w, lb_quat_w, kp2_lb)

        self.keypoints_command_w[:, 0:3] = kp0_cmd_w
        self.keypoints_command_w[:, 3:6] = kp1_cmd_w
        self.keypoints_command_w[:, 6:9] = kp2_cmd_w

        # Fixed goal in world
        goal_kp0_lb, goal_kp1_lb, goal_kp2_lb = self._split_kps(self.goal_keypoints_lb)
        goal_kp0_w = self._lb_points_to_world(lb_pos_w, lb_quat_w, goal_kp0_lb)
        goal_kp1_w = self._lb_points_to_world(lb_pos_w, lb_quat_w, goal_kp1_lb)
        goal_kp2_w = self._lb_points_to_world(lb_pos_w, lb_quat_w, goal_kp2_lb)

        self.goal_keypoints_w[:, 0:3] = goal_kp0_w
        self.goal_keypoints_w[:, 3:6] = goal_kp1_w
        self.goal_keypoints_w[:, 6:9] = goal_kp2_w

        # Current EE
        kp0_cur_w, kp1_cur_w, kp2_cur_w = self._current_keypoints_w()

        # Reference tracking metrics
        self.metrics["kp0_error"] = torch.linalg.norm(kp0_cmd_w - kp0_cur_w, dim=-1)
        self.metrics["kp1_error"] = torch.linalg.norm(kp1_cmd_w - kp1_cur_w, dim=-1)
        self.metrics["kp2_error"] = torch.linalg.norm(kp2_cmd_w - kp2_cur_w, dim=-1)
        self.metrics["position_error"].copy_(self.metrics["kp0_error"])

        # Goal tracking metrics
        self.metrics["goal_kp0_error"] = torch.linalg.norm(goal_kp0_w - kp0_cur_w, dim=-1)
        self.metrics["goal_kp1_error"] = torch.linalg.norm(goal_kp1_w - kp1_cur_w, dim=-1)
        self.metrics["goal_kp2_error"] = torch.linalg.norm(goal_kp2_w - kp2_cur_w, dim=-1)
        self.metrics["goal_position_error"].copy_(self.metrics["goal_kp0_error"])

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

        # Visualize fixed goal
        lb_pos_w, lb_quat_w = self._level_base_pose_w()
        goal_kp0_lb, goal_kp1_lb, goal_kp2_lb = self._split_kps(self.goal_keypoints_lb)

        goal_kp0_w = self._lb_points_to_world(lb_pos_w, lb_quat_w, goal_kp0_lb)
        goal_kp1_w = self._lb_points_to_world(lb_pos_w, lb_quat_w, goal_kp1_lb)
        goal_kp2_w = self._lb_points_to_world(lb_pos_w, lb_quat_w, goal_kp2_lb)

        goal_pts = torch.cat([goal_kp0_w, goal_kp1_w, goal_kp2_w], dim=0)
        self.kp_goal_vis.visualize(goal_pts)

        # Visualize current EE
        kp0_cur_w, kp1_cur_w, kp2_cur_w = self._current_keypoints_w()
        cur_pts = torch.cat([kp0_cur_w, kp1_cur_w, kp2_cur_w], dim=0)
        self.kp_cur_vis.visualize(cur_pts)


class PresampledKeypointsCubicTrajectoryCommandPLB(CommandTerm):
    """Cubic trajectory EE keypoint command in PLB frame.

    PLB frame:
      - origin = [base_x, base_y, ground_z]
      - orientation = yaw-only(base_quat)

    Command format:
      [kp0(3), kp1(3), kp2(3)] in PLB frame.

    Main difference from PresampledKeypointsDirectCommandPLB:
      direct: sampled endpoint is assigned immediately.
      this : sampled endpoint is reached through a smooth pose-level cubic trajectory.
    """

    cfg: "PresampledKeypointsCubicTrajectoryCommandPLBCfg"

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        path = cfg.file_path
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)

        arr = np.load(path).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 9:
            raise ValueError(
                f"[PresampledKeypointsCubicTrajectoryCommandPLB] "
                f"Expected npy shape (N, 9), got {arr.shape} from '{path}'."
            )

        self._table = torch.from_numpy(arr).to(self.device)
        self._num_rows = int(self._table.shape[0])

        if getattr(self.cfg, "sample_mode", "random") != "random":
            raise ValueError(
                "[PresampledKeypointsCubicTrajectoryCommandPLB] "
                "Only sample_mode='random' is supported."
            )

        self._dx = float(getattr(cfg, "kp_dx", 0.30))
        self._dz = float(getattr(cfg, "kp_dz", 0.30))
        self._ground_z = float(getattr(cfg, "ground_z", 0.0))

        self._off_x = torch.tensor([self._dx, 0.0, 0.0], device=self.device, dtype=torch.float32)
        self._off_z = torch.tensor([0.0, 0.0, self._dz], device=self.device, dtype=torch.float32)

        self._cycle_duration_s = float(getattr(cfg, "cycle_duration_s", 8.0))
        self._max_lin_vel = float(getattr(cfg, "max_lin_vel", 0.12))
        self._traj_duration_min_s = float(getattr(cfg, "traj_duration_min_s", 1.0))
        self._traj_duration_max_s = float(getattr(cfg, "traj_duration_max_s", 7.0))

        if self._cycle_duration_s <= 0.0:
            raise ValueError(f"Invalid cycle_duration_s={self._cycle_duration_s}")
        if self._max_lin_vel <= 0.0:
            raise ValueError(f"Invalid max_lin_vel={self._max_lin_vel}")
        if self._traj_duration_min_s <= 0.0:
            raise ValueError(f"Invalid traj_duration_min_s={self._traj_duration_min_s}")
        if self._traj_duration_max_s < self._traj_duration_min_s:
            raise ValueError(
                f"Invalid traj duration range: "
                f"({self._traj_duration_min_s}, {self._traj_duration_max_s})"
            )
        if self._traj_duration_max_s > self._cycle_duration_s:
            raise ValueError(
                f"traj_duration_max_s={self._traj_duration_max_s} "
                f"exceeds cycle_duration_s={self._cycle_duration_s}"
            )

        if hasattr(cfg, "resampling_time_range"):
            low, high = cfg.resampling_time_range
            if abs(float(low) - self._cycle_duration_s) > 1e-6 or abs(float(high) - self._cycle_duration_s) > 1e-6:
                raise ValueError(
                    "[PresampledKeypointsCubicTrajectoryCommandPLB] "
                    f"Expected resampling_time_range == "
                    f"({self._cycle_duration_s}, {self._cycle_duration_s}), "
                    f"got {cfg.resampling_time_range}."
                )

        box_min = torch.tensor(cfg.base_box_min, device=self.device, dtype=torch.float32)
        box_max = torch.tensor(cfg.base_box_max, device=self.device, dtype=torch.float32)
        margin = float(getattr(cfg, "base_box_margin", 0.08))

        self._base_box_min = box_min - margin
        self._base_box_max = box_max + margin

        self._collision_check_samples = int(getattr(cfg, "collision_check_samples", 20))
        self._resample_attempts = int(getattr(cfg, "resample_attempts", 50))
        self._collision_check_all_keypoints = bool(getattr(cfg, "collision_check_all_keypoints", True))
        self._reject_too_far_for_speed = bool(getattr(cfg, "reject_too_far_for_speed", True))

        # Current moving reference command
        self.keypoints_command_plb = torch.zeros(self.num_envs, 9, device=self.device)
        self.keypoints_command_w = torch.zeros_like(self.keypoints_command_plb)

        # Fixed sampled goal of the current cycle
        self.goal_keypoints_plb = torch.zeros(self.num_envs, 9, device=self.device)
        self.goal_keypoints_w = torch.zeros_like(self.goal_keypoints_plb)

        self._has_cmd = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Trajectory start pose in PLB
        self._traj_start_pos_plb = torch.zeros(self.num_envs, 3, device=self.device)
        self._traj_start_quat_plb = torch.zeros(self.num_envs, 4, device=self.device)
        self._traj_start_quat_plb[:, 0] = 1.0

        # Trajectory goal pose in PLB
        self._goal_pos_plb = torch.zeros(self.num_envs, 3, device=self.device)
        self._goal_quat_plb = torch.zeros(self.num_envs, 4, device=self.device)
        self._goal_quat_plb[:, 0] = 1.0

        # Per-env trajectory timing
        self._traj_duration_env = torch.full(
            (self.num_envs,),
            self._traj_duration_min_s,
            device=self.device,
            dtype=torch.float32,
        )
        self._hold_duration_env = torch.full(
            (self.num_envs,),
            self._cycle_duration_s - self._traj_duration_min_s,
            device=self.device,
            dtype=torch.float32,
        )

        # Metrics
        self.metrics["kp0_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["kp1_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["kp2_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)

        self.metrics["goal_kp0_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_kp1_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_kp2_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_position_error"] = torch.zeros(self.num_envs, device=self.device)

        self.metrics["traj_duration"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["hold_duration"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["traj_rejected"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "PresampledKeypointsCubicTrajectoryCommandPLB:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tTable rows: {self._num_rows}\n"
        msg += "\tSample mode: random\n"
        msg += "\tFrame: PLB projected level-base yaw-only frame\n"
        msg += f"\tground_z: {self._ground_z:.3f}\n"
        msg += f"\tKeypoints: kp0=pos, kp1=+X*{self._dx:.3f}, kp2=+Z*{self._dz:.3f}\n"
        msg += f"\tTiming: cycle={self._cycle_duration_s:.3f}s, max_lin_vel={self._max_lin_vel:.3f} m/s\n"
        msg += (
            f"\tTraj duration range: "
            f"({self._traj_duration_min_s:.3f}, {self._traj_duration_max_s:.3f}) s\n"
        )
        msg += (
            f"\tExpanded base box min={self._base_box_min.detach().cpu().numpy()}, "
            f"max={self._base_box_max.detach().cpu().numpy()}\n"
        )
        msg += "\tMode: pose-level cubic trajectory with base-box filtering\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        return self.keypoints_command_plb

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)

        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        self._has_cmd[env_ids_t] = False

        self.keypoints_command_plb[env_ids_t] = 0.0
        self.keypoints_command_w[env_ids_t] = 0.0
        self.goal_keypoints_plb[env_ids_t] = 0.0
        self.goal_keypoints_w[env_ids_t] = 0.0

        self._traj_start_pos_plb[env_ids_t] = 0.0
        self._traj_start_quat_plb[env_ids_t] = 0.0
        self._traj_start_quat_plb[env_ids_t, 0] = 1.0

        self._goal_pos_plb[env_ids_t] = 0.0
        self._goal_quat_plb[env_ids_t] = 0.0
        self._goal_quat_plb[env_ids_t, 0] = 1.0

        self._traj_duration_env[env_ids_t] = self._traj_duration_min_s
        self._hold_duration_env[env_ids_t] = self._cycle_duration_s - self._traj_duration_min_s

        self.metrics["traj_rejected"][env_ids_t] = 0.0

        self._resample_command(env_ids_t)

        return extras

    def _pick_indices(self, k: int) -> torch.Tensor:
        return torch.randint(0, self._num_rows, (k,), device=self.device)

    def _projected_level_base_pose_w(self) -> Tuple[torch.Tensor, torch.Tensor]:
        base_pos_w = self.robot.data.root_pos_w
        base_quat_w = self.robot.data.root_quat_w

        plb_pos_w = base_pos_w.clone()
        plb_pos_w[:, 2] = self._ground_z

        roll, pitch, yaw = euler_xyz_from_quat(base_quat_w)
        zeros = torch.zeros_like(yaw)
        plb_quat_w = quat_from_euler_xyz(zeros, zeros, yaw)

        return plb_pos_w, plb_quat_w

    def _plb_points_to_world(
        self,
        plb_pos_w: torch.Tensor,
        plb_quat_w: torch.Tensor,
        pts_plb: torch.Tensor,
    ) -> torch.Tensor:
        return plb_pos_w + quat_apply(plb_quat_w, pts_plb)

    def _world_points_to_plb(
        self,
        plb_pos_w: torch.Tensor,
        plb_quat_w: torch.Tensor,
        pts_w: torch.Tensor,
    ) -> torch.Tensor:
        plb_quat_inv_w = quat_inv(plb_quat_w)
        return quat_apply(plb_quat_inv_w, pts_w - plb_pos_w)

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

    def _current_keypoints_plb(self, env_ids_t: torch.Tensor) -> torch.Tensor:
        kp0_cur_w, kp1_cur_w, kp2_cur_w = self._current_keypoints_w()

        kp0_cur_w = kp0_cur_w[env_ids_t]
        kp1_cur_w = kp1_cur_w[env_ids_t]
        kp2_cur_w = kp2_cur_w[env_ids_t]

        plb_pos_w, plb_quat_w = self._projected_level_base_pose_w()
        plb_pos_w = plb_pos_w[env_ids_t]
        plb_quat_w = plb_quat_w[env_ids_t]

        kp0_cur_plb = self._world_points_to_plb(plb_pos_w, plb_quat_w, kp0_cur_w)
        kp1_cur_plb = self._world_points_to_plb(plb_pos_w, plb_quat_w, kp1_cur_w)
        kp2_cur_plb = self._world_points_to_plb(plb_pos_w, plb_quat_w, kp2_cur_w)

        return self._pack_kps(kp0_cur_plb, kp1_cur_plb, kp2_cur_plb)

    @staticmethod
    def _quat_normalize(q: torch.Tensor) -> torch.Tensor:
        return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _quat_slerp(q0: torch.Tensor, q1: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        q0 = PresampledKeypointsCubicTrajectoryCommandPLB._quat_normalize(q0)
        q1 = PresampledKeypointsCubicTrajectoryCommandPLB._quat_normalize(q1)

        dot = torch.sum(q0 * q1, dim=-1, keepdim=True)

        q1 = torch.where(dot < 0.0, -q1, q1)
        dot = torch.sum(q0 * q1, dim=-1, keepdim=True).clamp(-1.0, 1.0)

        s = s.unsqueeze(-1)

        theta_0 = torch.acos(dot)
        sin_theta_0 = torch.sin(theta_0)

        small = sin_theta_0.abs() < 1e-6

        theta = theta_0 * s
        sin_theta = torch.sin(theta)

        a = torch.sin(theta_0 - theta) / sin_theta_0.clamp_min(1e-8)
        b = sin_theta / sin_theta_0.clamp_min(1e-8)

        q = a * q0 + b * q1
        q_lerp = (1.0 - s) * q0 + s * q1

        q = torch.where(small, q_lerp, q)
        return PresampledKeypointsCubicTrajectoryCommandPLB._quat_normalize(q)

    @staticmethod
    def _quat_from_rotmat_wxyz(rot: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrix to quaternion in wxyz order.

        rot shape: (..., 3, 3)
        """
        m00 = rot[..., 0, 0]
        m01 = rot[..., 0, 1]
        m02 = rot[..., 0, 2]
        m10 = rot[..., 1, 0]
        m11 = rot[..., 1, 1]
        m12 = rot[..., 1, 2]
        m20 = rot[..., 2, 0]
        m21 = rot[..., 2, 1]
        m22 = rot[..., 2, 2]

        q = torch.zeros(rot.shape[:-2] + (4,), device=rot.device, dtype=rot.dtype)

        trace = m00 + m11 + m22

        cond = trace > 0.0

        s_trace = torch.sqrt((trace + 1.0).clamp_min(1e-8)) * 2.0
        qw_trace = 0.25 * s_trace
        qx_trace = (m21 - m12) / s_trace.clamp_min(1e-8)
        qy_trace = (m02 - m20) / s_trace.clamp_min(1e-8)
        qz_trace = (m10 - m01) / s_trace.clamp_min(1e-8)

        cond_x = (m00 > m11) & (m00 > m22) & (~cond)
        s_x = torch.sqrt((1.0 + m00 - m11 - m22).clamp_min(1e-8)) * 2.0
        qw_x = (m21 - m12) / s_x.clamp_min(1e-8)
        qx_x = 0.25 * s_x
        qy_x = (m01 + m10) / s_x.clamp_min(1e-8)
        qz_x = (m02 + m20) / s_x.clamp_min(1e-8)

        cond_y = (m11 > m22) & (~cond) & (~cond_x)
        s_y = torch.sqrt((1.0 + m11 - m00 - m22).clamp_min(1e-8)) * 2.0
        qw_y = (m02 - m20) / s_y.clamp_min(1e-8)
        qx_y = (m01 + m10) / s_y.clamp_min(1e-8)
        qy_y = 0.25 * s_y
        qz_y = (m12 + m21) / s_y.clamp_min(1e-8)

        cond_z = (~cond) & (~cond_x) & (~cond_y)
        s_z = torch.sqrt((1.0 + m22 - m00 - m11).clamp_min(1e-8)) * 2.0
        qw_z = (m10 - m01) / s_z.clamp_min(1e-8)
        qx_z = (m02 + m20) / s_z.clamp_min(1e-8)
        qy_z = (m12 + m21) / s_z.clamp_min(1e-8)
        qz_z = 0.25 * s_z

        q[..., 0] = torch.where(
            cond,
            qw_trace,
            torch.where(cond_x, qw_x, torch.where(cond_y, qw_y, qw_z)),
        )
        q[..., 1] = torch.where(
            cond,
            qx_trace,
            torch.where(cond_x, qx_x, torch.where(cond_y, qx_y, qx_z)),
        )
        q[..., 2] = torch.where(
            cond,
            qy_trace,
            torch.where(cond_x, qy_x, torch.where(cond_y, qy_y, qy_z)),
        )
        q[..., 3] = torch.where(
            cond,
            qz_trace,
            torch.where(cond_x, qz_x, torch.where(cond_y, qz_y, qz_z)),
        )

        return PresampledKeypointsCubicTrajectoryCommandPLB._quat_normalize(q)

    def _quat_from_keypoints_plb(
        self,
        kp0: torch.Tensor,
        kp1: torch.Tensor,
        kp2: torch.Tensor,
    ) -> torch.Tensor:
        x_axis = kp1 - kp0
        z_axis = kp2 - kp0

        x_axis = x_axis / torch.linalg.norm(x_axis, dim=-1, keepdim=True).clamp_min(1e-8)
        z_axis = z_axis / torch.linalg.norm(z_axis, dim=-1, keepdim=True).clamp_min(1e-8)

        y_axis = torch.cross(z_axis, x_axis, dim=-1)
        y_axis = y_axis / torch.linalg.norm(y_axis, dim=-1, keepdim=True).clamp_min(1e-8)

        z_axis = torch.cross(x_axis, y_axis, dim=-1)
        z_axis = z_axis / torch.linalg.norm(z_axis, dim=-1, keepdim=True).clamp_min(1e-8)

        rot = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        return self._quat_from_rotmat_wxyz(rot)

    @staticmethod
    def _cubic_time_scaling(tau: torch.Tensor) -> torch.Tensor:
        tau = tau.clamp(0.0, 1.0)
        return 3.0 * tau * tau - 2.0 * tau * tau * tau

    def _compute_traj_duration_from_distance(
        self,
        env_ids_t: torch.Tensor,
        kp0_start: torch.Tensor,
        kp0_goal: torch.Tensor,
    ) -> torch.Tensor:
        dist = torch.linalg.norm(kp0_goal - kp0_start, dim=-1)

        traj = dist / self._max_lin_vel
        traj = traj.clamp(self._traj_duration_min_s, self._traj_duration_max_s)
        hold = self._cycle_duration_s - traj

        self._traj_duration_env[env_ids_t] = traj
        self._hold_duration_env[env_ids_t] = hold

        self.metrics["traj_duration"][env_ids_t] = traj
        self.metrics["hold_duration"][env_ids_t] = hold

        return traj

    def _points_inside_base_box(self, pts: torch.Tensor) -> torch.Tensor:
        """Check whether points are inside expanded base box.

        pts shape:
          (..., 3)

        returns:
          (...) bool
        """
        inside_min = pts >= self._base_box_min
        inside_max = pts <= self._base_box_max
        return torch.all(inside_min & inside_max, dim=-1)

    def _trajectory_collides_base(
        self,
        kp0_start: torch.Tensor,
        quat_start: torch.Tensor,
        kp0_goal: torch.Tensor,
        quat_goal: torch.Tensor,
    ) -> torch.Tensor:
        """Check whether intermediate trajectory enters expanded base box.

        Important:
          This intentionally skips tau=0 and tau=1.
          Endpoints are assumed to be from the collision-free presampled set.
        """
        num = kp0_start.shape[0]
        collides = torch.zeros(num, device=self.device, dtype=torch.bool)

        n = max(2, self._collision_check_samples)

        for i in range(1, n):
            tau = torch.full(
                (num,),
                float(i) / float(n),
                device=self.device,
                dtype=torch.float32,
            )
            s = self._cubic_time_scaling(tau)

            pos = kp0_start + s.unsqueeze(-1) * (kp0_goal - kp0_start)
            quat = self._quat_slerp(quat_start, quat_goal, s)

            kp1, kp2 = self._kps_from_pose(pos, quat)

            if self._collision_check_all_keypoints:
                inside = (
                    self._points_inside_base_box(pos)
                    | self._points_inside_base_box(kp1)
                    | self._points_inside_base_box(kp2)
                )
            else:
                inside = self._points_inside_base_box(pos)

            collides |= inside

        return collides

    def _eval_current_traj_command(self, env_ids_t: torch.Tensor) -> torch.Tensor:
        t_left = self.time_left[env_ids_t]
        t = (self._cycle_duration_s - t_left).clamp(min=0.0, max=self._cycle_duration_s)

        traj_duration = self._traj_duration_env[env_ids_t].clamp_min(1e-6)
        tau = (t / traj_duration).clamp(0.0, 1.0)
        s = self._cubic_time_scaling(tau)

        start_pos = self._traj_start_pos_plb[env_ids_t]
        start_quat = self._traj_start_quat_plb[env_ids_t]
        goal_pos = self._goal_pos_plb[env_ids_t]
        goal_quat = self._goal_quat_plb[env_ids_t]

        pos = start_pos + s.unsqueeze(-1) * (goal_pos - start_pos)
        quat = self._quat_slerp(start_quat, goal_quat, s)

        kp1, kp2 = self._kps_from_pose(pos, quat)
        return self._pack_kps(pos, kp1, kp2)

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        k = env_ids_t.numel()
        if k == 0:
            return

        has_prev = self._has_cmd[env_ids_t]
        kps_prev_cmd = self.keypoints_command_plb[env_ids_t].clone()
        kps_cur_plb = self._current_keypoints_plb(env_ids_t)

        # Start from previous command if available. Otherwise start from current measured EE pose.
        kps_start = torch.where(has_prev.unsqueeze(-1), kps_prev_cmd, kps_cur_plb)

        kp0_start, kp1_start, kp2_start = self._split_kps(kps_start)
        quat_start = self._quat_from_keypoints_plb(kp0_start, kp1_start, kp2_start)

        accepted = torch.zeros(k, device=self.device, dtype=torch.bool)

        kp0_goal_final = kp0_start.clone()
        quat_goal_final = quat_start.clone()
        goal_kps_final = kps_start.clone()

        rejected_count = torch.zeros(k, device=self.device, dtype=torch.float32)

        max_dist_allowed = self._max_lin_vel * self._traj_duration_max_s

        for _ in range(self._resample_attempts):
            remaining = torch.nonzero(~accepted, as_tuple=False).squeeze(-1)
            if remaining.numel() == 0:
                break

            m = remaining.numel()

            idx = self._pick_indices(m)
            candidate_kps = self._table[idx].clone()

            kp0_cand, kp1_cand, kp2_cand = self._split_kps(candidate_kps)
            quat_cand = self._quat_from_keypoints_plb(kp0_cand, kp1_cand, kp2_cand)

            kp0_start_rem = kp0_start[remaining]
            quat_start_rem = quat_start[remaining]

            dist = torch.linalg.norm(kp0_cand - kp0_start_rem, dim=-1)

            if self._reject_too_far_for_speed:
                speed_ok = dist <= max_dist_allowed
            else:
                speed_ok = torch.ones(m, device=self.device, dtype=torch.bool)

            collides = self._trajectory_collides_base(
                kp0_start=kp0_start_rem,
                quat_start=quat_start_rem,
                kp0_goal=kp0_cand,
                quat_goal=quat_cand,
            )

            ok = speed_ok & (~collides)

            rejected_count[remaining] += (~ok).float()

            if ok.any():
                ok_local = torch.nonzero(ok, as_tuple=False).squeeze(-1)
                global_local = remaining[ok_local]

                accepted[global_local] = True

                kp0_goal_final[global_local] = kp0_cand[ok_local]
                quat_goal_final[global_local] = quat_cand[ok_local]
                goal_kps_final[global_local] = candidate_kps[ok_local]

        # Fallback: if no safe target is found, hold current command.
        # This avoids injecting unsafe commands when the base box is too conservative.
        not_accepted = torch.nonzero(~accepted, as_tuple=False).squeeze(-1)
        if not_accepted.numel() > 0:
            kp0_goal_final[not_accepted] = kp0_start[not_accepted]
            quat_goal_final[not_accepted] = quat_start[not_accepted]
            goal_kps_final[not_accepted] = kps_start[not_accepted]

        self.metrics["traj_rejected"][env_ids_t] = rejected_count

        self._compute_traj_duration_from_distance(
            env_ids_t=env_ids_t,
            kp0_start=kp0_start,
            kp0_goal=kp0_goal_final,
        )

        self._traj_start_pos_plb[env_ids_t] = kp0_start
        self._traj_start_quat_plb[env_ids_t] = quat_start

        self._goal_pos_plb[env_ids_t] = kp0_goal_final
        self._goal_quat_plb[env_ids_t] = quat_goal_final

        self.goal_keypoints_plb[env_ids_t] = goal_kps_final

        # Initialize current reference at start of trajectory.
        kp1_start_rebuild, kp2_start_rebuild = self._kps_from_pose(kp0_start, quat_start)
        self.keypoints_command_plb[env_ids_t] = self._pack_kps(
            kp0_start,
            kp1_start_rebuild,
            kp2_start_rebuild,
        )

        self._has_cmd[env_ids_t] = True

    def _update_command(self):
        valid_env_ids = torch.nonzero(self._has_cmd, as_tuple=False).squeeze(-1)
        if valid_env_ids.numel() == 0:
            return

        self.keypoints_command_plb[valid_env_ids] = self._eval_current_traj_command(valid_env_ids)

    def _update_metrics(self):
        plb_pos_w, plb_quat_w = self._projected_level_base_pose_w()

        # Current moving reference in world
        kp0_plb, kp1_plb, kp2_plb = self._split_kps(self.keypoints_command_plb)

        kp0_cmd_w = self._plb_points_to_world(plb_pos_w, plb_quat_w, kp0_plb)
        kp1_cmd_w = self._plb_points_to_world(plb_pos_w, plb_quat_w, kp1_plb)
        kp2_cmd_w = self._plb_points_to_world(plb_pos_w, plb_quat_w, kp2_plb)

        self.keypoints_command_w[:, 0:3] = kp0_cmd_w
        self.keypoints_command_w[:, 3:6] = kp1_cmd_w
        self.keypoints_command_w[:, 6:9] = kp2_cmd_w

        # Fixed goal in world
        goal_kp0_plb, goal_kp1_plb, goal_kp2_plb = self._split_kps(self.goal_keypoints_plb)

        goal_kp0_w = self._plb_points_to_world(plb_pos_w, plb_quat_w, goal_kp0_plb)
        goal_kp1_w = self._plb_points_to_world(plb_pos_w, plb_quat_w, goal_kp1_plb)
        goal_kp2_w = self._plb_points_to_world(plb_pos_w, plb_quat_w, goal_kp2_plb)

        self.goal_keypoints_w[:, 0:3] = goal_kp0_w
        self.goal_keypoints_w[:, 3:6] = goal_kp1_w
        self.goal_keypoints_w[:, 6:9] = goal_kp2_w

        # Current actual EE keypoints
        kp0_cur_w, kp1_cur_w, kp2_cur_w = self._current_keypoints_w()

        # Moving reference tracking metrics
        self.metrics["kp0_error"] = torch.linalg.norm(kp0_cmd_w - kp0_cur_w, dim=-1)
        self.metrics["kp1_error"] = torch.linalg.norm(kp1_cmd_w - kp1_cur_w, dim=-1)
        self.metrics["kp2_error"] = torch.linalg.norm(kp2_cmd_w - kp2_cur_w, dim=-1)
        self.metrics["position_error"].copy_(self.metrics["kp0_error"])

        # Final goal tracking metrics
        self.metrics["goal_kp0_error"] = torch.linalg.norm(goal_kp0_w - kp0_cur_w, dim=-1)
        self.metrics["goal_kp1_error"] = torch.linalg.norm(goal_kp1_w - kp1_cur_w, dim=-1)
        self.metrics["goal_kp2_error"] = torch.linalg.norm(goal_kp2_w - kp2_cur_w, dim=-1)
        self.metrics["goal_position_error"].copy_(self.metrics["goal_kp0_error"])

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

        plb_pos_w, plb_quat_w = self._projected_level_base_pose_w()

        # Visualize fixed final goal
        goal_kp0_plb, goal_kp1_plb, goal_kp2_plb = self._split_kps(self.goal_keypoints_plb)

        goal_kp0_w = self._plb_points_to_world(plb_pos_w, plb_quat_w, goal_kp0_plb)
        goal_kp1_w = self._plb_points_to_world(plb_pos_w, plb_quat_w, goal_kp1_plb)
        goal_kp2_w = self._plb_points_to_world(plb_pos_w, plb_quat_w, goal_kp2_plb)

        goal_pts = torch.cat([goal_kp0_w, goal_kp1_w, goal_kp2_w], dim=0)
        self.kp_goal_vis.visualize(goal_pts)

        # Visualize actual current EE
        kp0_cur_w, kp1_cur_w, kp2_cur_w = self._current_keypoints_w()
        cur_pts = torch.cat([kp0_cur_w, kp1_cur_w, kp2_cur_w], dim=0)
        self.kp_cur_vis.visualize(cur_pts)