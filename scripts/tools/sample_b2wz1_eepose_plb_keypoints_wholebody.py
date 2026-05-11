"""
Sample collision-free EE keypoints for Unitree B2W+Z1 in Projected Level-Base frame.

Design:
- free root
- joint targets held during settling
- all envs get one of three front-leg posture modes:
    0) default stand      : default_ratio
    1) front-leg squat   : squat_ratio, helps low / near-ground EE poses
    2) front-leg straight: straight_ratio, helps high EE poses

Workspace quota:
- x > front_x_threshold:
    target ratio = front_x_ratio, default 80%
    keep existing low/high-z logic:
        low-z front samples: kp0_z <= low_z_threshold
        low-z ratio inside front bucket = low_z_ratio
- x < front_x_threshold:
    target ratio = 1 - front_x_ratio, default 20%
    no z_min / low_z_threshold / z_bias_low filtering
    accept collision-free, joint-limit-safe, orientation-valid samples directly

PLB frame:
- origin = [base_x, base_y, ground_z]
- orientation = yaw-only(base_quat)

Usage:
./isaaclab.sh -p scripts/tools/sample_b2wz1_eepose_plb_keypoints_wholebody.py \
  --out scripts/tools/reachable_kp0kp1kp2_plb_wholebody_v2.npy \
  --num_samples 50000 \
  --front_x_ratio 0.80 \
  --low_z_ratio 0.30 \
  --low_z_threshold 0.50 \
  --headless
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()

parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--num_samples", type=int, default=50000)
parser.add_argument(
    "--out",
    type=str,
    default="scripts/tools/reachable_kp0kp1kp2_plb_wholebody_v2.npy",
)

# posture mixture
# posture_id:
#   0 = default stand
#   1 = front-leg squat
#   2 = front-leg straight / extend
parser.add_argument("--default_ratio", type=float, default=0.50)
parser.add_argument("--squat_ratio", type=float, default=0.35)
parser.add_argument("--straight_ratio", type=float, default=0.15)

parser.add_argument("--front_thigh_default", type=float, default=0.80)
parser.add_argument("--front_calf_default", type=float, default=-1.50)

# Squat branch: lower EE / near-ground reach.
parser.add_argument("--front_thigh_squat", type=float, default=1.00)
parser.add_argument("--front_calf_squat", type=float, default=-2.50)

# Straight branch: higher EE reach. Keep mild by default.
parser.add_argument("--front_thigh_straight", type=float, default=0.65)
parser.add_argument("--front_calf_straight", type=float, default=-1.15)

parser.add_argument("--hip_noise_std", type=float, default=0.015)
parser.add_argument("--thigh_noise_std", type=float, default=0.015)
parser.add_argument("--calf_noise_std", type=float, default=0.025)

# PLB
parser.add_argument("--ground_z", type=float, default=0.0)

# Workspace quota
parser.add_argument("--front_x_threshold", type=float, default=0.0)
parser.add_argument("--front_x_ratio", type=float, default=0.80)

# Optional hard lower bound for x<0 samples.
# This prevents accepting extremely far-back poses if they appear.
# Set to a very negative value if you truly want no x lower bound.
parser.add_argument("--back_x_min", type=float, default=-0.35)

# Front-region low-z distribution control.
# Applied only when kp0_x > front_x_threshold.
parser.add_argument("--z_min", type=float, default=0.035)
parser.add_argument("--low_z_threshold", type=float, default=0.50)
parser.add_argument("--low_z_ratio", type=float, default=0.30)

# sim / collision
parser.add_argument("--settle_steps", type=int, default=600)
parser.add_argument("--force_threshold", type=float, default=3.0)

# joints
parser.add_argument("--arm_joints", type=str, default="joint1,joint2,joint3,joint4,joint5,joint6")
parser.add_argument(
    "--leg_joints",
    type=str,
    default=(
        "FL_hip_joint,FL_thigh_joint,FL_calf_joint,FL_foot_joint,"
        "FR_hip_joint,FR_thigh_joint,FR_calf_joint,FR_foot_joint,"
        "RL_hip_joint,RL_thigh_joint,RL_calf_joint,RL_foot_joint,"
        "RR_hip_joint,RR_thigh_joint,RR_calf_joint,RR_foot_joint"
    ),
)
parser.add_argument("--joint_margin_deg", type=float, default=10.0)

# spatial diversity
parser.add_argument("--voxel_size", type=float, default=0.035)
parser.add_argument("--max_per_voxel_front", type=int, default=8)
parser.add_argument("--max_per_voxel_back", type=int, default=8)

# orientation constraints
parser.add_argument("--yaw_min_deg", type=float, default=-90.0)
parser.add_argument("--yaw_max_deg", type=float, default=90.0)
parser.add_argument("--pitch_x_min_deg", type=float, default=-90.0)
parser.add_argument("--pitch_x_max_deg", type=float, default=90.0)
parser.add_argument("--require_z_up", action="store_true")
parser.set_defaults(require_z_up=True)

# Acceptance bias for x>0 front region only.
# For x<0 region, this is not applied.
parser.add_argument("--x_bias", type=float, default=0.0)
parser.add_argument("--x0", type=float, default=0.25)
parser.add_argument("--x_sigma", type=float, default=0.15)
parser.add_argument("--x_floor", type=float, default=0.25)

# Low-z shaping for x>0 squat samples only.
# This is not applied to x<0 samples.
parser.add_argument("--z_bias_low", type=float, default=0.75)
parser.add_argument("--z0", type=float, default=0.22)
parser.add_argument("--z_sigma", type=float, default=0.10)
parser.add_argument("--z_floor", type=float, default=0.20)

# keypoints
parser.add_argument("--kp_dx", type=float, default=0.30)
parser.add_argument("--kp_dz", type=float, default=0.30)

# quality
parser.add_argument("--ortho_tol", type=float, default=2e-3)
parser.add_argument("--norm_tol", type=float, default=3e-3)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import os
import math
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import UNITREE_B2WZ1_HIGHGAINS_CFG


# Do not include wheels/legs since ground contact is normal.
# Do not include gripper, otherwise near-ground grasp poses may be rejected.
CONTACT_REGEX = "(base_link|link00|link01|link02|link03|link04|link05|link06)"

Z1_LIMITS_DEG = {
    "joint1": (-150.0, 150.0),
    "joint2": (0.0, 180.0),
    "joint3": (-165.0, 0.0),
    "joint4": (-80.0, 80.0),
    "joint5": (-85.0, 85.0),
    "joint6": (-160.0, 160.0),
}

STAND_LEG_POS = {
    "FL_hip_joint": 0.1,
    "FR_hip_joint": -0.1,
    "RL_hip_joint": 0.1,
    "RR_hip_joint": -0.1,
    "FL_thigh_joint": 0.8,
    "FR_thigh_joint": 0.8,
    "RL_thigh_joint": 1.0,
    "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5,
    "FR_calf_joint": -1.5,
    "RL_calf_joint": -1.5,
    "RR_calf_joint": -1.5,
    "FL_foot_joint": 0.0,
    "FR_foot_joint": 0.0,
    "RL_foot_joint": 0.0,
    "RR_foot_joint": 0.0,
}


def _deg2rad(x: float) -> float:
    return float(x) * math.pi / 180.0


@configclass
class SampleSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot: ArticulationCfg = UNITREE_B2WZ1_HIGHGAINS_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UNITREE_B2WZ1_HIGHGAINS_CFG.spawn.replace(
            articulation_props=UNITREE_B2WZ1_HIGHGAINS_CFG.spawn.articulation_props.replace(
                fix_root_link=False,
            )
        ),
    )

    contact_forces = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{CONTACT_REGEX}",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )


def yaw_only_quat(q_wxyz: torch.Tensor) -> torch.Tensor:
    _, _, yaw = math_utils.euler_xyz_from_quat(q_wxyz)
    zeros = torch.zeros_like(yaw)
    return math_utils.quat_from_euler_xyz(zeros, zeros, yaw)


def _resolve_ids(robot, arm_joint_names, leg_joint_names, ee_body_name: str):
    arm_joint_ids, _ = robot.find_joints(arm_joint_names)
    leg_joint_ids, leg_joint_resolved_names = robot.find_joints(leg_joint_names)
    ee_body_ids, _ = robot.find_bodies(ee_body_name)
    return arm_joint_ids, leg_joint_ids, leg_joint_resolved_names, int(ee_body_ids[0])


def _get_joint_limits_rad(robot, joint_names, joint_ids: torch.Tensor):
    lim = robot.data.soft_joint_pos_limits
    low = lim[0, joint_ids, 0].clone()
    high = lim[0, joint_ids, 1].clone()

    for i, name in enumerate(joint_names):
        if name in Z1_LIMITS_DEG:
            low[i] = _deg2rad(Z1_LIMITS_DEG[name][0])
            high[i] = _deg2rad(Z1_LIMITS_DEG[name][1])

    return low, high


def _sample_arm_q(robot, arm_joint_names, arm_joint_ids: torch.Tensor):
    low, high = _get_joint_limits_rad(robot, arm_joint_names, arm_joint_ids)
    num_envs = robot.data.joint_pos.shape[0]
    num_joints = arm_joint_ids.numel()

    q_uniform = low.unsqueeze(0) + torch.rand((num_envs, num_joints), device=robot.device) * (
        high - low
    ).unsqueeze(0)

    q_ref = torch.tensor([0.0, 120.0, -120.0, 0.0, 0.0, 0.0], device=robot.device) * math.pi / 180.0
    q_low = torch.tensor([0.0, 145.0, -150.0, -30.0, 0.0, 0.0], device=robot.device) * math.pi / 180.0

    std_ref = torch.tensor([25.0, 18.0, 18.0, 30.0, 30.0, 35.0], device=robot.device) * math.pi / 180.0
    std_low = torch.tensor([20.0, 12.0, 12.0, 20.0, 25.0, 30.0], device=robot.device) * math.pi / 180.0

    q_ref_sample = q_ref.unsqueeze(0) + torch.randn((num_envs, num_joints), device=robot.device) * std_ref.unsqueeze(0)
    q_low_sample = q_low.unsqueeze(0) + torch.randn((num_envs, num_joints), device=robot.device) * std_low.unsqueeze(0)

    q_ref_sample = torch.max(torch.min(q_ref_sample, high.unsqueeze(0)), low.unsqueeze(0))
    q_low_sample = torch.max(torch.min(q_low_sample, high.unsqueeze(0)), low.unsqueeze(0))

    r = torch.rand((num_envs, 1), device=robot.device)
    use_ref = r < 0.20
    use_low = (r >= 0.20) & (r < 0.60)

    q = q_uniform
    q = torch.where(use_ref.expand(-1, num_joints), q_ref_sample, q)
    q = torch.where(use_low.expand(-1, num_joints), q_low_sample, q)
    return q


def _ok_not_near_joint_limits(robot, joint_pos, joint_names, joint_ids, margin_deg: float):
    margin = _deg2rad(margin_deg)
    low, high = _get_joint_limits_rad(robot, joint_names, joint_ids)
    q = joint_pos[:, joint_ids]
    min_margin = torch.minimum(q - low.unsqueeze(0), high.unsqueeze(0) - q).min(dim=1).values
    return min_margin > margin


def _make_leg_targets(robot, leg_joint_ids, leg_joint_names):
    dtype = robot.data.default_joint_pos.dtype
    device = robot.device
    num_envs = robot.data.default_joint_pos.shape[0]

    leg_q = robot.data.default_joint_pos[:, leg_joint_ids].clone()

    for j, name in enumerate(leg_joint_names):
        if name in STAND_LEG_POS:
            leg_q[:, j] = float(STAND_LEG_POS[name])

    default_ratio = float(np.clip(args_cli.default_ratio, 0.0, 1.0))
    squat_ratio = float(np.clip(args_cli.squat_ratio, 0.0, 1.0))
    straight_ratio = float(np.clip(args_cli.straight_ratio, 0.0, 1.0))

    ratio_sum = max(default_ratio + squat_ratio + straight_ratio, 1e-8)
    default_ratio /= ratio_sum
    squat_ratio /= ratio_sum
    straight_ratio /= ratio_sum

    r = torch.rand((num_envs,), device=device, dtype=dtype)
    is_default = r < default_ratio
    is_squat = (r >= default_ratio) & (r < default_ratio + squat_ratio)
    is_straight = r >= default_ratio + squat_ratio

    posture_id = torch.zeros((num_envs,), device=device, dtype=torch.long)
    posture_id[is_default] = 0
    posture_id[is_squat] = 1
    posture_id[is_straight] = 2

    alpha = torch.zeros((num_envs,), device=device, dtype=dtype)
    n_squat = int(is_squat.sum().item())
    n_straight = int(is_straight.sum().item())
    if n_squat > 0:
        alpha[is_squat] = torch.rand((n_squat,), device=device, dtype=dtype)
    if n_straight > 0:
        alpha[is_straight] = torch.rand((n_straight,), device=device, dtype=dtype)

    thigh_default = float(args_cli.front_thigh_default)
    calf_default = float(args_cli.front_calf_default)

    thigh = torch.full((num_envs,), thigh_default, device=device, dtype=dtype)
    calf = torch.full((num_envs,), calf_default, device=device, dtype=dtype)

    thigh_squat = thigh_default + (float(args_cli.front_thigh_squat) - thigh_default) * alpha
    calf_squat = calf_default + (float(args_cli.front_calf_squat) - calf_default) * alpha

    thigh_straight = thigh_default + (float(args_cli.front_thigh_straight) - thigh_default) * alpha
    calf_straight = calf_default + (float(args_cli.front_calf_straight) - calf_default) * alpha

    thigh[is_squat] = thigh_squat[is_squat]
    calf[is_squat] = calf_squat[is_squat]
    thigh[is_straight] = thigh_straight[is_straight]
    calf[is_straight] = calf_straight[is_straight]

    name_to_col = {name: j for j, name in enumerate(leg_joint_names)}

    def set_val(name: str, value):
        if name in name_to_col:
            col = name_to_col[name]
            if isinstance(value, torch.Tensor):
                leg_q[:, col] = value
            else:
                leg_q[:, col] = float(value)

    # Front legs: default / squat / straight branch.
    set_val("FL_hip_joint", 0.1)
    set_val("FR_hip_joint", -0.1)
    set_val("FL_thigh_joint", thigh)
    set_val("FR_thigh_joint", thigh)
    set_val("FL_calf_joint", calf)
    set_val("FR_calf_joint", calf)
    set_val("FL_foot_joint", 0.0)
    set_val("FR_foot_joint", 0.0)

    # Rear legs stay at stand default.
    set_val("RL_hip_joint", 0.1)
    set_val("RR_hip_joint", -0.1)
    set_val("RL_thigh_joint", 1.0)
    set_val("RR_thigh_joint", 1.0)
    set_val("RL_calf_joint", -1.5)
    set_val("RR_calf_joint", -1.5)
    set_val("RL_foot_joint", 0.0)
    set_val("RR_foot_joint", 0.0)

    # Add small noise only to non-default branches to keep default samples clean.
    non_default = ~is_default
    if int(non_default.sum().item()) > 0:
        hip_noise = torch.randn((num_envs,), device=device, dtype=dtype) * float(args_cli.hip_noise_std)
        thigh_noise = torch.randn((num_envs,), device=device, dtype=dtype) * float(args_cli.thigh_noise_std)
        calf_noise = torch.randn((num_envs,), device=device, dtype=dtype) * float(args_cli.calf_noise_std)

        if "FL_hip_joint" in name_to_col:
            leg_q[non_default, name_to_col["FL_hip_joint"]] += hip_noise[non_default]
        if "FR_hip_joint" in name_to_col:
            leg_q[non_default, name_to_col["FR_hip_joint"]] -= hip_noise[non_default]

        for name in ["FL_thigh_joint", "FR_thigh_joint"]:
            if name in name_to_col:
                leg_q[non_default, name_to_col[name]] += thigh_noise[non_default]

        for name in ["FL_calf_joint", "FR_calf_joint"]:
            if name in name_to_col:
                leg_q[non_default, name_to_col[name]] += calf_noise[non_default]

    return leg_q, alpha, posture_id


def _make_joint_targets(robot, arm_joint_ids, leg_joint_ids, leg_joint_names, q_arm):
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()

    joint_pos[:, arm_joint_ids] = q_arm
    joint_vel[:, arm_joint_ids] = 0.0

    leg_q, alpha, posture_id = _make_leg_targets(robot, leg_joint_ids, leg_joint_names)
    joint_pos[:, leg_joint_ids] = leg_q
    joint_vel[:, leg_joint_ids] = 0.0

    return joint_pos, joint_vel, alpha, posture_id


def _reset_envs(scene, arm_joint_ids, leg_joint_ids, leg_joint_names, q_arm):
    robot = scene["robot"]
    scene.reset()

    joint_pos, joint_vel, alpha, posture_id = _make_joint_targets(
        robot, arm_joint_ids, leg_joint_ids, leg_joint_names, q_arm
    )

    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins

    robot.write_root_state_to_sim(root_state)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    robot.set_joint_position_target(joint_pos)
    try:
        robot.set_joint_velocity_target(torch.zeros_like(joint_vel))
    except AttributeError:
        pass

    return joint_pos, joint_vel, alpha, posture_id


def _hold_targets_and_step(scene, sim, sim_dt, joint_pos, joint_vel, steps: int):
    robot = scene["robot"]
    zero_vel = torch.zeros_like(joint_vel)

    for _ in range(steps):
        robot.set_joint_position_target(joint_pos)
        try:
            robot.set_joint_velocity_target(zero_vel)
        except AttributeError:
            pass

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)


def _voxel_index(pos: np.ndarray, voxel_size: float):
    return tuple(np.floor(pos / voxel_size).astype(np.int32).tolist())


def _sigmoid(x: float) -> float:
    if x >= 0:
        e = np.exp(-x)
        return float(1.0 / (1.0 + e))
    e = np.exp(x)
    return float(e / (1.0 + e))


def _accept_prob_x(x: float, x_bias: float, x0: float, x_sigma: float, x_floor: float):
    s = _sigmoid((x - x0) / max(1e-6, x_sigma))
    shaped = x_floor + (1.0 - x_floor) * s
    return float(np.clip((1.0 - x_bias) + x_bias * shaped, 1e-6, 1.0))


def _accept_prob_low_z(z: float, z_bias: float, z0: float, z_sigma: float, z_floor: float):
    s = _sigmoid((z0 - z) / max(1e-6, z_sigma))
    shaped = z_floor + (1.0 - z_floor) * s
    return float(np.clip((1.0 - z_bias) + z_bias * shaped, 1e-6, 1.0))


def _orthogonality_ok(kp0, kp1, kp2, dx, dz, ortho_tol, norm_tol):
    v1 = kp1 - kp0
    v2 = kp2 - kp0
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))

    if n1 < 1e-9 or n2 < 1e-9:
        return False
    if abs(n1 - dx) > norm_tol:
        return False
    if abs(n2 - dz) > norm_tol:
        return False

    cos = float(np.dot(v1, v2) / (n1 * n2))
    return abs(cos) <= ortho_tol


@torch.no_grad()
def main():
    device = args_cli.device

    arm_joint_names = [s.strip() for s in args_cli.arm_joints.split(",")]
    leg_joint_names = [s.strip() for s in args_cli.leg_joints.split(",")]
    ee_body_name = "gripperStator"

    target_total = int(args_cli.num_samples)

    front_x_ratio = float(np.clip(args_cli.front_x_ratio, 0.0, 1.0))
    target_front = int(math.ceil(target_total * front_x_ratio))
    target_back = target_total - target_front

    target_front_low = int(math.ceil(target_front * float(args_cli.low_z_ratio)))
    target_front_high = target_front - target_front_low

    dx = float(args_cli.kp_dx)
    dz = float(args_cli.kp_dz)

    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.0, 2.0, 2.2], target=[0.0, 0.0, 0.6])

    scene_cfg = SampleSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    sim_dt = sim.get_physics_dt()
    scene.reset()
    scene.update(sim_dt)

    robot = scene["robot"]

    arm_joint_ids, leg_joint_ids, leg_joint_resolved_names, ee_body_id = _resolve_ids(
        robot, arm_joint_names, leg_joint_names, ee_body_name
    )

    if not isinstance(arm_joint_ids, torch.Tensor):
        arm_joint_ids = torch.tensor(arm_joint_ids, dtype=torch.long, device=robot.device)
    if not isinstance(leg_joint_ids, torch.Tensor):
        leg_joint_ids = torch.tensor(leg_joint_ids, dtype=torch.long, device=robot.device)

    leg_joint_names = list(leg_joint_resolved_names)

    print("[INFO] target total:", target_total)
    print("[INFO] target front x>threshold:", target_front, f"threshold={args_cli.front_x_threshold}")
    print("[INFO] target back  x<threshold:", target_back, f"back_x_min={args_cli.back_x_min}")
    print("[INFO] target front low-z:", target_front_low, f"z <= {args_cli.low_z_threshold}")
    print("[INFO] target front high-z:", target_front_high)
    print("[INFO] front_x_ratio:", args_cli.front_x_ratio)
    print("[INFO] front low_z_ratio:", args_cli.low_z_ratio)
    print("[INFO] default_ratio:", args_cli.default_ratio)
    print("[INFO] squat_ratio:", args_cli.squat_ratio)
    print("[INFO] straight_ratio:", args_cli.straight_ratio)
    print("[INFO] front default thigh/calf:", args_cli.front_thigh_default, args_cli.front_calf_default)
    print("[INFO] front squat thigh/calf:", args_cli.front_thigh_squat, args_cli.front_calf_squat)
    print("[INFO] front straight thigh/calf:", args_cli.front_thigh_straight, args_cli.front_calf_straight)
    print("[INFO] front z_min:", args_cli.z_min)
    print("[INFO] settle_steps:", args_cli.settle_steps, f"({args_cli.settle_steps * sim_dt:.2f}s)")
    print("[INFO] arm joints:", arm_joint_names)
    print("[INFO] arm joint ids:", arm_joint_ids.detach().cpu().tolist())
    print("[INFO] leg joints:", leg_joint_names)
    print("[INFO] leg joint ids:", leg_joint_ids.detach().cpu().tolist())
    print("[INFO] EE body:", ee_body_name, ee_body_id)
    print("[INFO] PLB ground_z:", args_cli.ground_z)
    print("[INFO] contact regex:", CONTACT_REGEX)

    ex = torch.tensor([1.0, 0.0, 0.0], device=robot.device)
    ez = torch.tensor([0.0, 0.0, 1.0], device=robot.device)

    yaw_min = float(args_cli.yaw_min_deg)
    yaw_max = float(args_cli.yaw_max_deg)
    pitch_x_min = float(args_cli.pitch_x_min_deg)
    pitch_x_max = float(args_cli.pitch_x_max_deg)
    deg = 180.0 / math.pi

    rng = np.random.default_rng()

    collected_front_low = []
    collected_front_high = []
    collected_back = []

    posture_front_low = []
    posture_front_high = []
    posture_back = []

    voxel_front_low = {}
    voxel_front_high = {}
    voxel_back = {}

    it = 0

    while (
        len(collected_front_low) < target_front_low
        or len(collected_front_high) < target_front_high
        or len(collected_back) < target_back
    ) and simulation_app.is_running():
        it += 1

        num_envs = robot.data.joint_pos.shape[0]
        q_arm = _sample_arm_q(robot, arm_joint_names, arm_joint_ids)

        joint_pos_target, joint_vel_target, alpha, posture_id = _reset_envs(
            scene, arm_joint_ids, leg_joint_ids, leg_joint_names, q_arm
        )

        _hold_targets_and_step(
            scene, sim, sim_dt, joint_pos_target, joint_vel_target, int(args_cli.settle_steps)
        )

        contact = scene["contact_forces"]
        forces_w = contact.data.net_forces_w
        force_norm = torch.linalg.norm(forces_w, dim=-1)
        collided = (force_norm > float(args_cli.force_threshold)).any(dim=1)

        ok = ~collided
        ok = ok & _ok_not_near_joint_limits(
            robot, robot.data.joint_pos, arm_joint_names, arm_joint_ids, float(args_cli.joint_margin_deg)
        )

        base_pos_w = robot.data.root_pos_w
        base_quat_w = robot.data.root_quat_w

        plb_pos_w = base_pos_w.clone()
        plb_pos_w[:, 2] = float(args_cli.ground_z)
        plb_quat_w = yaw_only_quat(base_quat_w)
        plb_quat_inv = math_utils.quat_conjugate(plb_quat_w)

        ee_pos_w = robot.data.body_pos_w[:, ee_body_id, :]
        ee_quat_w = robot.data.body_quat_w[:, ee_body_id, :]

        ee_pos_plb = math_utils.quat_apply_inverse(plb_quat_w, ee_pos_w - plb_pos_w)
        ee_quat_plb = math_utils.quat_mul(plb_quat_inv, ee_quat_w)
        ee_quat_plb = math_utils.quat_unique(ee_quat_plb)

        x_ee_plb = math_utils.quat_apply(ee_quat_plb, ex.unsqueeze(0).expand_as(ee_pos_plb))
        z_ee_plb = math_utils.quat_apply(ee_quat_plb, ez.unsqueeze(0).expand_as(ee_pos_plb))

        yaw_x = torch.atan2(x_ee_plb[:, 1], x_ee_plb[:, 0]) * deg
        pitch_x = torch.atan2(
            x_ee_plb[:, 2],
            torch.sqrt(x_ee_plb[:, 0] ** 2 + x_ee_plb[:, 1] ** 2) + 1e-9,
        ) * deg

        ok = ok & (yaw_x >= yaw_min) & (yaw_x <= yaw_max)
        ok = ok & (pitch_x >= pitch_x_min) & (pitch_x <= pitch_x_max)

        if bool(args_cli.require_z_up):
            ok = ok & (z_ee_plb[:, 2] > 0.0)

        kp0 = ee_pos_plb
        off_x = torch.tensor([dx, 0.0, 0.0], device=robot.device).unsqueeze(0).expand_as(kp0)
        off_z = torch.tensor([0.0, 0.0, dz], device=robot.device).unsqueeze(0).expand_as(kp0)

        kp1 = kp0 + math_utils.quat_apply(ee_quat_plb, off_x)
        kp2 = kp0 + math_utils.quat_apply(ee_quat_plb, off_z)
        keypoints = torch.cat([kp0, kp1, kp2], dim=-1)

        ok_np = ok.detach().cpu().numpy()
        keypoints_np = keypoints.detach().cpu().numpy().astype(np.float32)
        pos_np = kp0.detach().cpu().numpy().astype(np.float32)
        posture_np = posture_id.detach().cpu().numpy().astype(np.int64)

        accepted_front_low = 0
        accepted_front_high = 0
        accepted_back = 0

        for env_i in np.nonzero(ok_np)[0]:
            row_kp = keypoints_np[env_i]
            row_pos = pos_np[env_i]

            x = float(row_pos[0])
            z = float(row_pos[2])
            posture = int(posture_np[env_i])

            front_threshold = float(args_cli.front_x_threshold)

            # x > 0 region: preserve old front low/high-z logic.
            if x > front_threshold:
                # For front workspace, keep z_min and low-z quota logic.
                if z < float(args_cli.z_min):
                    continue

                is_low_z = z <= float(args_cli.low_z_threshold)

                if is_low_z:
                    if len(collected_front_low) >= target_front_low:
                        continue
                    collected = collected_front_low
                    collected_posture = posture_front_low
                    voxel = voxel_front_low
                    max_per_voxel = int(args_cli.max_per_voxel_front)
                else:
                    if len(collected_front_high) >= target_front_high:
                        continue
                    collected = collected_front_high
                    collected_posture = posture_front_high
                    voxel = voxel_front_high
                    max_per_voxel = int(args_cli.max_per_voxel_front)

                # Existing x acceptance shaping, applied only to x>0.
                if float(args_cli.x_bias) > 0.0:
                    p_x = _accept_prob_x(
                        x,
                        float(args_cli.x_bias),
                        float(args_cli.x0),
                        float(args_cli.x_sigma),
                        float(args_cli.x_floor),
                    )
                    if rng.random() > p_x:
                        continue

                # Existing low-z shaping, applied only to front squat samples.
                if posture == 1:
                    p_z = _accept_prob_low_z(
                        z,
                        float(args_cli.z_bias_low),
                        float(args_cli.z0),
                        float(args_cli.z_sigma),
                        float(args_cli.z_floor),
                    )
                    if rng.random() > p_z:
                        continue

                vidx = _voxel_index(row_pos, float(args_cli.voxel_size))
                c = voxel.get(vidx, 0)
                if c >= max_per_voxel:
                    continue

                voxel[vidx] = c + 1
                collected.append(row_kp)
                collected_posture.append(posture)

                if is_low_z:
                    accepted_front_low += 1
                else:
                    accepted_front_high += 1

            # x < 0 region: no z filtering, no low/high-z split, no z bias.
            elif x < front_threshold:
                if len(collected_back) >= target_back:
                    continue

                if x < float(args_cli.back_x_min):
                    continue

                vidx = _voxel_index(row_pos, float(args_cli.voxel_size))
                c = voxel_back.get(vidx, 0)
                if c >= int(args_cli.max_per_voxel_back):
                    continue

                voxel_back[vidx] = c + 1
                collected_back.append(row_kp)
                posture_back.append(posture)
                accepted_back += 1

            # Exactly x == threshold is rare; skip to keep strict x>0 / x<0 definition.
            else:
                continue

        if it % 10 == 0:
            ok_count = int(ok.sum().item())
            ok_z_min = float(ee_pos_plb[ok, 2].min()) if ok_count > 0 else float("nan")
            ok_z_max = float(ee_pos_plb[ok, 2].max()) if ok_count > 0 else float("nan")
            ok_x_min = float(ee_pos_plb[ok, 0].min()) if ok_count > 0 else float("nan")
            ok_x_max = float(ee_pos_plb[ok, 0].max()) if ok_count > 0 else float("nan")

            n_default = int((posture_id == 0).sum().item())
            n_squat = int((posture_id == 1).sum().item())
            n_straight = int((posture_id == 2).sum().item())

            print(
                f"[INFO] iter={it} "
                f"front_low={len(collected_front_low)}/{target_front_low} "
                f"front_high={len(collected_front_high)}/{target_front_high} "
                f"back={len(collected_back)}/{target_back} "
                f"accepted=[fl:{accepted_front_low}, fh:{accepted_front_high}, b:{accepted_back}] "
                f"ok={ok_count}/{num_envs} "
                f"posture=[default:{n_default}, squat:{n_squat}, straight:{n_straight}] "
                f"ok_x=[{ok_x_min:.3f},{ok_x_max:.3f}] "
                f"raw_z=[{float(ee_pos_plb[:, 2].min()):.3f},{float(ee_pos_plb[:, 2].max()):.3f}] "
                f"ok_z=[{ok_z_min:.3f},{ok_z_max:.3f}] "
                f"base_z_mean={float(base_pos_w[:, 2].mean()):.3f}"
            )

    if len(collected_front_low) + len(collected_front_high) + len(collected_back) == 0:
        print("[WARN] no samples collected.")
        sim.stop()
        return

    front_low = np.asarray(collected_front_low, dtype=np.float32)
    front_high = np.asarray(collected_front_high, dtype=np.float32)
    back = np.asarray(collected_back, dtype=np.float32)

    out_parts = []
    posture_parts = []

    if front_low.shape[0] > 0:
        out_parts.append(front_low)
        posture_parts.append(np.asarray(posture_front_low, dtype=np.int64))
    if front_high.shape[0] > 0:
        out_parts.append(front_high)
        posture_parts.append(np.asarray(posture_front_high, dtype=np.int64))
    if back.shape[0] > 0:
        out_parts.append(back)
        posture_parts.append(np.asarray(posture_back, dtype=np.int64))

    out_all = np.concatenate(out_parts, axis=0)
    posture_all = np.concatenate(posture_parts, axis=0)

    keep = []
    for i in range(out_all.shape[0]):
        k0 = out_all[i, 0:3]
        k1 = out_all[i, 3:6]
        k2 = out_all[i, 6:9]

        if _orthogonality_ok(
            k0,
            k1,
            k2,
            dx=dx,
            dz=dz,
            ortho_tol=float(args_cli.ortho_tol),
            norm_tol=float(args_cli.norm_tol),
        ):
            keep.append(i)

    keep = np.asarray(keep, dtype=np.int64)
    out = out_all[keep]
    posture_out = posture_all[keep]

    perm = rng.permutation(out.shape[0])
    out = out[perm]
    posture_out = posture_out[perm]

    out_path = args_cli.out
    out_dir = os.path.dirname(out_path)

    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    np.save(out_path, out)

    posture_path = out_path.replace(".npy", "_posture_id.npy")
    np.save(posture_path, posture_out)

    workspace_id = np.zeros((out.shape[0],), dtype=np.int64)
    workspace_id[out[:, 0] > float(args_cli.front_x_threshold)] = 1
    workspace_id[out[:, 0] < float(args_cli.front_x_threshold)] = -1

    workspace_path = out_path.replace(".npy", "_workspace_id.npy")
    np.save(workspace_path, workspace_id)

    meta = {
        "front_x_threshold": float(args_cli.front_x_threshold),
        "front_x_ratio": float(args_cli.front_x_ratio),
        "back_x_min": float(args_cli.back_x_min),
        "z_min_front_only": float(args_cli.z_min),
        "low_z_threshold_front_only": float(args_cli.low_z_threshold),
        "low_z_ratio_front_only": float(args_cli.low_z_ratio),
        "default_ratio": float(args_cli.default_ratio),
        "squat_ratio": float(args_cli.squat_ratio),
        "straight_ratio": float(args_cli.straight_ratio),
        "kp_dx": float(args_cli.kp_dx),
        "kp_dz": float(args_cli.kp_dz),
        "require_z_up": bool(args_cli.require_z_up),
    }
    meta_path = out_path.replace(".npy", "_meta.npy")
    np.save(meta_path, meta)

    front_final = out[out[:, 0] > float(args_cli.front_x_threshold)]
    back_final = out[out[:, 0] < float(args_cli.front_x_threshold)]
    front_low_final = front_final[front_final[:, 2] <= float(args_cli.low_z_threshold)]

    n0 = int((posture_out == 0).sum())
    n1 = int((posture_out == 1).sum())
    n2 = int((posture_out == 2).sum())
    total_final = max(1, int(out.shape[0]))

    print(f"[DONE] saved: {out_path}")
    print(f"[DONE] saved posture ids: {posture_path}")
    print(f"[DONE] saved workspace ids: {workspace_path}")
    print(f"[DONE] saved meta: {meta_path}")
    print(f"[DONE] collected before QC: front_low={len(collected_front_low)}, front_high={len(collected_front_high)}, back={len(collected_back)}")
    print(f"[DONE] final shape: {out.shape}")
    print(
        "[DONE] final workspace composition: "
        f"front_x>threshold={front_final.shape[0]} ({front_final.shape[0] / total_final:.3f}), "
        f"back_x<threshold={back_final.shape[0]} ({back_final.shape[0] / total_final:.3f})"
    )
    if front_final.shape[0] > 0:
        print(
            "[DONE] final front low-z composition: "
            f"{front_low_final.shape[0]} / {front_final.shape[0]} "
            f"({front_low_final.shape[0] / max(1, front_final.shape[0]):.3f})"
        )

    print(
        "[DONE] final posture composition: "
        f"default={n0} ({n0 / total_final:.3f}), "
        f"squat={n1} ({n1 / total_final:.3f}), "
        f"straight={n2} ({n2 / total_final:.3f})"
    )
    print("[STATS] kp0 x min/max/mean:", float(out[:, 0].min()), float(out[:, 0].max()), float(out[:, 0].mean()))
    print("[STATS] kp0 y min/max/mean:", float(out[:, 1].min()), float(out[:, 1].max()), float(out[:, 1].mean()))
    print("[STATS] kp0 z min/max/mean:", float(out[:, 2].min()), float(out[:, 2].max()), float(out[:, 2].mean()))
    print("[CHECK] mean ||kp1-kp0||:", float(np.mean(np.linalg.norm(out[:, 3:6] - out[:, 0:3], axis=1))))
    print("[CHECK] mean ||kp2-kp0||:", float(np.mean(np.linalg.norm(out[:, 6:9] - out[:, 0:3], axis=1))))

    sim.stop()


if __name__ == "__main__":
    main()
    simulation_app.close()