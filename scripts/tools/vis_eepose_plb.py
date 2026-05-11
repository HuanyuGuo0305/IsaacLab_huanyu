"""
Visualize sampled EE poses as points in world frame.

This script:
- Spawns Unitree B2W+Z1
- Loads keypoints in Projected Level-Base frame, PLB
- Converts PLB points -> World and draws them as debug points

PLB frame:
- origin = [base_x, base_y, ground_z]
- orientation = yaw-only(base_quat)

Usage:
./isaaclab.sh -p scripts/tools/vis_eepose_plb.py \
  --npy scripts/tools/reachable_kp0kp1kp2_plb.npy
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--npy", type=str, default="reachable_kp0kp1kp2_plb.npy")
parser.add_argument("--max_points", type=int, default=10000)
parser.add_argument("--point_size", type=float, default=3.0)
parser.add_argument("--base_z", type=float, default=None)
parser.add_argument("--ground_z", type=float, default=0.0)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import UNITREE_B2WZ1_CFG

import isaacsim.util.debug_draw._debug_draw as omni_debug_draw


@configclass
class VisSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot: ArticulationCfg = UNITREE_B2WZ1_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )


def yaw_only_quat(q_wxyz_batched: torch.Tensor) -> torch.Tensor:
    _, _, yaw = math_utils.euler_xyz_from_quat(q_wxyz_batched)
    zeros = torch.zeros_like(yaw)
    return math_utils.quat_from_euler_xyz(zeros, zeros, yaw)


@torch.no_grad()
def main():
    keypoints_plb = np.load(args_cli.npy).astype(np.float32)
    n_total = keypoints_plb.shape[0]

    pos_plb = keypoints_plb[:, 0:3]

    max_points = int(args_cli.max_points)
    if n_total > max_points:
        idx = np.random.choice(n_total, size=max_points, replace=False)
        pos_plb = pos_plb[idx]

    n_vis = pos_plb.shape[0]
    pos_plb_t = torch.from_numpy(pos_plb)

    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.0, 1.8], [0.0, 0.0, 0.7])

    scene_cfg = VisSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    sim_dt = sim.get_physics_dt()

    scene.reset()
    scene.update(sim_dt)

    robot = scene["robot"]

    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins

    if args_cli.base_z is not None:
        root_state[:, 2] = float(args_cli.base_z)

    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]))

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    scene.write_data_to_sim()
    sim.step()
    scene.update(sim_dt)

    base_pos_w = robot.data.root_pos_w
    base_quat_w = robot.data.root_quat_w

    print(f"[INFO] Base height world z: {float(base_pos_w[0, 2]):.4f} m")
    print(f"[INFO] PLB ground_z: {float(args_cli.ground_z):.4f} m")

    root_pose_frozen = torch.cat(
        [robot.data.root_pos_w, robot.data.root_quat_w],
        dim=-1,
    ).clone()
    root_vel_frozen = torch.zeros_like(root_state[:, 7:])
    joint_pos_frozen = joint_pos.clone()
    joint_vel_frozen = torch.zeros_like(joint_vel)

    # PLB -> World
    # PLB origin: projected base origin on ground plane
    plb_pos_w = base_pos_w.clone()
    plb_pos_w[:, 2] = float(args_cli.ground_z)

    # PLB orientation: yaw-only base orientation
    plb_quat_w = yaw_only_quat(base_quat_w)

    pos_plb_dev = pos_plb_t.to(device=robot.device)

    q = plb_quat_w.repeat(pos_plb_dev.shape[0], 1)
    p0 = plb_pos_w.repeat(pos_plb_dev.shape[0], 1)

    pos_w = p0 + math_utils.quat_apply(q, pos_plb_dev)

    dd = omni_debug_draw.acquire_debug_draw_interface()
    dd.clear_points()

    pts = pos_w.detach().cpu().tolist()
    colors = [(1.0, 0.0, 0.0, 1.0)] * len(pts)
    sizes = [float(args_cli.point_size)] * len(pts)

    dd.draw_points(pts, colors, sizes)

    print(f"[DONE] Drew {n_vis}/{n_total} PLB points from {args_cli.npy}")

    while simulation_app.is_running():
        robot.write_root_pose_to_sim(root_pose_frozen)
        robot.write_root_velocity_to_sim(root_vel_frozen)
        robot.write_joint_state_to_sim(joint_pos_frozen, joint_vel_frozen)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    sim.stop()


if __name__ == "__main__":
    main()
    simulation_app.close()