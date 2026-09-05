"""
Sample collision-free EE keypoints for Unitree B2W+Z1 in Projected Level-Base frame.

Design:
- free root
- joint targets held during settling
- two front-leg posture modes:
    0) default stand
    1) front-leg squat
- no front-leg straight branch.

Sampling quotas, collected simultaneously from mixed-posture batches:
- task ground bucket: 10% of total samples:
    kp0_z in [0.06, 0.15]
    EE local +X pitch in [task_pitch_x_min_deg, task_pitch_x_max_deg]
    EE local +X yaw in [task_yaw_x_min_deg, task_yaw_x_max_deg]
    EE roll in [task_roll_min_deg, task_roll_max_deg]
    all samples use maximum squat posture alpha=1.0
- mid-height bucket: 40% of total samples:
    kp0_z in [0.15, 0.60]
    samples use uniform interpolation from default front-leg posture to max squat posture
- default bucket: 50% of total samples:
    samples use default posture and accept all collision-free, joint-limit-safe,
    orientation-valid keypoints. No x/y/z bucket filtering is applied.

Important body convention:
- The low-level policy tracks gripperStator keypoints.
- Therefore all saved kp0/kp1/kp2 are computed from gripperStator.
- High-level pickup may use a virtual gripperCenter offset, but this sampler filters by gripperStator kp0.

PLB frame:
- origin = [base_x, base_y, ground_z]
- orientation = yaw-only(base_quat)

Orientation filter:
- Global yaw/pitch/roll filters are kept broad for workspace diversity.
- The task ground bucket applies stricter pickup-oriented yaw and roll filters.

Usage:
./isaaclab.sh -p scripts/tools/sample_b2wz1_eepose_plb_keypoints_wholebody_pickup.py \
  --out scripts/tools/keypoints_plb_pickup.npy \
  --num_samples 25000 \
  --task_bucket_ratio 0.10 \
  --mid_z_ratio 0.40 \
  --headless
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()

parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--num_samples", type=int, default=30000)
parser.add_argument(
    "--out",
    type=str,
    default="scripts/tools/keypoints_plb_pickup.npy",
)


parser.add_argument("--front_thigh_default", type=float, default=0.80)
parser.add_argument("--front_calf_default", type=float, default=-1.50)

# Squat branch: lower EE / near-ground reach.
parser.add_argument("--front_thigh_squat", type=float, default=1.00)
parser.add_argument("--front_calf_squat", type=float, default=-2.50)



# PLB
parser.add_argument("--ground_z", type=float, default=0.0)


# Task and height bucket quotas.
parser.add_argument("--task_bucket_ratio", type=float, default=0.10)
parser.add_argument("--task_z_min", type=float, default=0.06)
parser.add_argument("--task_z_max", type=float, default=0.15)
# Applied only to the task bucket. pitch_x is the elevation of EE local +X axis in PLB frame.
parser.add_argument("--task_pitch_x_min_deg", type=float, default=-75.0)
parser.add_argument("--task_pitch_x_max_deg", type=float, default=0.0)
# Applied only to the task bucket. yaw_x is the heading of EE local +X axis in PLB frame.
# This enforces pickup-oriented local +X forward samples without restricting default workspace diversity.
parser.add_argument("--task_yaw_x_min_deg", type=float, default=-60.0)
parser.add_argument("--task_yaw_x_max_deg", type=float, default=60.0)
# Applied only to the task bucket.  This keeps the pickup bucket near the new
# joint6=90deg wrist/gripper orientation while leaving global roll broad.
parser.add_argument("--task_roll_min_deg", type=float, default=30.0)
parser.add_argument("--task_roll_max_deg", type=float, default=150.0)
parser.add_argument("--mid_z_ratio", type=float, default=0.40)
parser.add_argument("--mid_z_min", type=float, default=0.15)
parser.add_argument("--mid_z_max", type=float, default=0.60)


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
parser.add_argument("--max_per_voxel_task", type=int, default=8)
parser.add_argument("--max_per_voxel_mid", type=int, default=8)
parser.add_argument("--max_per_voxel_front", type=int, default=8)

# orientation constraints
parser.add_argument("--yaw_min_deg", type=float, default=-90.0)
parser.add_argument("--yaw_max_deg", type=float, default=90.0)
parser.add_argument("--pitch_x_min_deg", type=float, default=-75.0)
parser.add_argument("--pitch_x_max_deg", type=float, default=30.0)
parser.add_argument("--roll_min_deg", type=float, default=-150.0)
parser.add_argument("--roll_max_deg", type=float, default=150.0)

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

from isaaclab_assets.robots.unitree import UNITREE_B2WZ1_SAMPLING_CFG


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

    robot: ArticulationCfg = UNITREE_B2WZ1_SAMPLING_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UNITREE_B2WZ1_SAMPLING_CFG.spawn.replace(
            articulation_props=UNITREE_B2WZ1_SAMPLING_CFG.spawn.articulation_props.replace(
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

    # Bias pickup-oriented arm samples toward a 90deg wrist orientation.
    # The last entry is joint6; the other arm joints keep the previous reference centers.
    q_ref = torch.tensor([0.0, 120.0, -120.0, 0.0, 0.0, 90.0], device=robot.device) * math.pi / 180.0
    q_low = torch.tensor([0.0, 145.0, -150.0, -30.0, 0.0, 90.0], device=robot.device) * math.pi / 180.0

    # Keep joint6 somewhat concentrated around 90deg, while preserving enough spread
    # for low-level tracking robustness.
    std_ref = torch.tensor([25.0, 18.0, 18.0, 30.0, 30.0, 20.0], device=robot.device) * math.pi / 180.0
    std_low = torch.tensor([20.0, 12.0, 12.0, 20.0, 25.0, 20.0], device=robot.device) * math.pi / 180.0

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


def _make_leg_targets(robot, leg_joint_ids, leg_joint_names, balanced_probs):
    dtype = robot.data.default_joint_pos.dtype
    device = robot.device
    num_envs = robot.data.default_joint_pos.shape[0]

    leg_q = robot.data.default_joint_pos[:, leg_joint_ids].clone()
    for j, name in enumerate(leg_joint_names):
        if name in STAND_LEG_POS:
            leg_q[:, j] = float(STAND_LEG_POS[name])

    p_task, p_mid, p_default = [float(max(0.0, x)) for x in balanced_probs]
    p_sum = max(p_task + p_mid + p_default, 1.0e-8)
    p_task, p_mid = p_task / p_sum, p_mid / p_sum

    r = torch.rand((num_envs,), device=device, dtype=dtype)
    is_task_max_squat = r < p_task
    is_mid_squat = (r >= p_task) & (r < p_task + p_mid)
    is_squat = is_task_max_squat | is_mid_squat

    posture_id = is_squat.to(dtype=torch.long)
    alpha = torch.zeros((num_envs,), device=device, dtype=dtype)
    alpha[is_task_max_squat] = 1.0
    n_mid = int(is_mid_squat.sum().item())
    if n_mid > 0:
        alpha[is_mid_squat] = torch.rand((n_mid,), device=device, dtype=dtype)

    thigh_default = float(args_cli.front_thigh_default)
    calf_default = float(args_cli.front_calf_default)
    thigh = torch.full((num_envs,), thigh_default, device=device, dtype=dtype)
    calf = torch.full((num_envs,), calf_default, device=device, dtype=dtype)
    thigh[is_squat] = thigh_default + (float(args_cli.front_thigh_squat) - thigh_default) * alpha[is_squat]
    calf[is_squat] = calf_default + (float(args_cli.front_calf_squat) - calf_default) * alpha[is_squat]

    name_to_col = {name: j for j, name in enumerate(leg_joint_names)}

    def set_val(name: str, value):
        if name in name_to_col:
            leg_q[:, name_to_col[name]] = value

    set_val("FL_hip_joint", 0.1)
    set_val("FR_hip_joint", -0.1)
    set_val("FL_thigh_joint", thigh)
    set_val("FR_thigh_joint", thigh)
    set_val("FL_calf_joint", calf)
    set_val("FR_calf_joint", calf)
    set_val("FL_foot_joint", 0.0)
    set_val("FR_foot_joint", 0.0)
    set_val("RL_hip_joint", 0.1)
    set_val("RR_hip_joint", -0.1)
    set_val("RL_thigh_joint", 1.0)
    set_val("RR_thigh_joint", 1.0)
    set_val("RL_calf_joint", -1.5)
    set_val("RR_calf_joint", -1.5)
    set_val("RL_foot_joint", 0.0)
    set_val("RR_foot_joint", 0.0)

    return leg_q, alpha, posture_id


def _make_joint_targets(robot, arm_joint_ids, leg_joint_ids, leg_joint_names, q_arm, balanced_probs):
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()

    joint_pos[:, arm_joint_ids] = q_arm
    joint_vel[:, arm_joint_ids] = 0.0

    leg_q, alpha, posture_id = _make_leg_targets(
        robot, leg_joint_ids, leg_joint_names, balanced_probs
    )
    joint_pos[:, leg_joint_ids] = leg_q
    joint_vel[:, leg_joint_ids] = 0.0

    return joint_pos, joint_vel, alpha, posture_id


def _reset_envs(scene, arm_joint_ids, leg_joint_ids, leg_joint_names, q_arm, balanced_probs):
    robot = scene["robot"]
    scene.reset()

    joint_pos, joint_vel, alpha, posture_id = _make_joint_targets(
        robot, arm_joint_ids, leg_joint_ids, leg_joint_names, q_arm, balanced_probs
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
    task_bucket_ratio = float(np.clip(args_cli.task_bucket_ratio, 0.0, 1.0))
    mid_z_ratio = float(np.clip(args_cli.mid_z_ratio, 0.0, 1.0))

    target_task = int(round(target_total * task_bucket_ratio))
    target_mid = int(round(target_total * mid_z_ratio))
    target_default = target_total - target_task - target_mid
    if target_default < 0:
        raise ValueError(
            "Invalid quota: task_bucket_ratio + mid_z_ratio exceeds 1. "
            f"Got task={task_bucket_ratio}, mid={mid_z_ratio}."
        )

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
    print(
        "[INFO] target task ground bucket:",
        target_task,
        f"z=[{args_cli.task_z_min},{args_cli.task_z_max}], "
        f"pitch_x=[{args_cli.task_pitch_x_min_deg},{args_cli.task_pitch_x_max_deg}], "
        f"yaw_x=[{args_cli.task_yaw_x_min_deg},{args_cli.task_yaw_x_max_deg}], "
        f"roll=[{args_cli.task_roll_min_deg},{args_cli.task_roll_max_deg}], "
        "posture=max_squat",
    )
    print(
        "[INFO] target mid-height bucket:",
        target_mid,
        f"z=[{args_cli.mid_z_min},{args_cli.mid_z_max}], posture=uniform default->max_squat",
    )
    print("[INFO] target default bucket:", target_default, "posture=default, accept all valid samples")
    print("[INFO] default front thigh/calf:", args_cli.front_thigh_default, args_cli.front_calf_default)
    print("[INFO] max squat front thigh/calf:", args_cli.front_thigh_squat, args_cli.front_calf_squat)
    print("[INFO] settle_steps:", args_cli.settle_steps, f"({args_cli.settle_steps * sim_dt:.2f}s)")
    print("[INFO] arm joints:", arm_joint_names)
    print("[INFO] arm joint ids:", arm_joint_ids.detach().cpu().tolist())
    print("[INFO] leg joints:", leg_joint_names)
    print("[INFO] leg joint ids:", leg_joint_ids.detach().cpu().tolist())
    print("[INFO] EE body for saved keypoints:", ee_body_name, ee_body_id)
    print("[INFO] PLB ground_z:", args_cli.ground_z)
    print("[INFO] global yaw range:", args_cli.yaw_min_deg, args_cli.yaw_max_deg)
    print("[INFO] global pitch_x range:", args_cli.pitch_x_min_deg, args_cli.pitch_x_max_deg)
    print("[INFO] global roll range:", args_cli.roll_min_deg, args_cli.roll_max_deg)
    print("[INFO] contact regex:", CONTACT_REGEX)

    ex = torch.tensor([1.0, 0.0, 0.0], device=robot.device)

    yaw_min = float(args_cli.yaw_min_deg)
    yaw_max = float(args_cli.yaw_max_deg)
    pitch_x_min = float(args_cli.pitch_x_min_deg)
    pitch_x_max = float(args_cli.pitch_x_max_deg)
    roll_min = float(args_cli.roll_min_deg)
    roll_max = float(args_cli.roll_max_deg)
    deg = 180.0 / math.pi

    rng = np.random.default_rng()

    collected_task = []
    collected_mid = []
    collected_default = []

    voxel_task = {}
    voxel_mid = {}
    voxel_default = {}

    it = 0

    while (
        len(collected_task) < target_task
        or len(collected_mid) < target_mid
        or len(collected_default) < target_default
    ) and simulation_app.is_running():
        it += 1

        remaining_task = max(target_task - len(collected_task), 0)
        remaining_mid = max(target_mid - len(collected_mid), 0)
        remaining_default = max(target_default - len(collected_default), 0)
        remaining_total = max(remaining_task + remaining_mid + remaining_default, 1)

        # Mixed-posture batch: every iteration contains candidate envs for all unfinished buckets.
        # This avoids the previous sequential behavior where the sampler filled task first,
        # then mid, then default.
        balanced_probs = (
            remaining_task / remaining_total,
            remaining_mid / remaining_total,
            remaining_default / remaining_total,
        )
        num_envs = robot.data.joint_pos.shape[0]
        q_arm = _sample_arm_q(robot, arm_joint_names, arm_joint_ids)

        joint_pos_target, joint_vel_target, alpha, posture_id = _reset_envs(
            scene,
            arm_joint_ids,
            leg_joint_ids,
            leg_joint_names,
            q_arm,
            balanced_probs,
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
        yaw_x = torch.atan2(x_ee_plb[:, 1], x_ee_plb[:, 0]) * deg
        pitch_x = torch.atan2(
            x_ee_plb[:, 2],
            torch.sqrt(x_ee_plb[:, 0] ** 2 + x_ee_plb[:, 1] ** 2) + 1e-9,
        ) * deg

        roll = math_utils.euler_xyz_from_quat(ee_quat_plb)[0] * deg

        ok = ok & (yaw_x >= yaw_min) & (yaw_x <= yaw_max)
        ok = ok & (pitch_x >= pitch_x_min) & (pitch_x <= pitch_x_max)
        ok = ok & (roll >= roll_min) & (roll <= roll_max)

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
        alpha_np = alpha.detach().cpu().numpy().astype(np.float32)
        yaw_x_np = yaw_x.detach().cpu().numpy().astype(np.float32)
        pitch_x_np = pitch_x.detach().cpu().numpy().astype(np.float32)
        roll_np = roll.detach().cpu().numpy().astype(np.float32)

        accepted_task = 0
        accepted_mid = 0
        accepted_default = 0

        for env_i in np.nonzero(ok_np)[0]:
            row_kp = keypoints_np[env_i]
            row_pos = pos_np[env_i]
            z = float(row_pos[2])
            yaw_i = float(yaw_x_np[env_i])
            pitch_i = float(pitch_x_np[env_i])
            roll_i = float(roll_np[env_i])
            posture = int(posture_np[env_i])
            alpha_i = float(alpha_np[env_i])

            # Task bucket: strict pickup-oriented samples.
            # Require max-squat, low height, local +X roughly forward, and
            # wrist/EE roll near the new joint6=90deg grasp orientation.
            in_task_bucket = (
                z >= float(args_cli.task_z_min)
                and z <= float(args_cli.task_z_max)
                and pitch_i >= float(args_cli.task_pitch_x_min_deg)
                and pitch_i <= float(args_cli.task_pitch_x_max_deg)
                and yaw_i >= float(args_cli.task_yaw_x_min_deg)
                and yaw_i <= float(args_cli.task_yaw_x_max_deg)
                and roll_i >= float(args_cli.task_roll_min_deg)
                and roll_i <= float(args_cli.task_roll_max_deg)
                and posture == 1
                and alpha_i >= 0.999
            )
            if in_task_bucket and len(collected_task) < target_task:
                vidx = _voxel_index(row_pos, float(args_cli.voxel_size))
                c = voxel_task.get(vidx, 0)
                if c < int(args_cli.max_per_voxel_task):
                    voxel_task[vidx] = c + 1
                    collected_task.append(row_kp)
                    accepted_task += 1
                    continue

            # Mid-height bucket: squat interpolation alpha in [0, 1), not exact max-squat.
            in_mid_bucket = (
                z >= float(args_cli.mid_z_min)
                and z <= float(args_cli.mid_z_max)
                and posture == 1
                and alpha_i < 0.999
            )
            if in_mid_bucket and len(collected_mid) < target_mid:
                vidx = _voxel_index(row_pos, float(args_cli.voxel_size))
                c = voxel_mid.get(vidx, 0)
                if c < int(args_cli.max_per_voxel_mid):
                    voxel_mid[vidx] = c + 1
                    collected_mid.append(row_kp)
                    accepted_mid += 1
                    continue

            # Default bucket: default posture, all collision-free / joint-safe / orientation-valid samples.
            if posture == 0 and len(collected_default) < target_default:
                vidx = _voxel_index(row_pos, float(args_cli.voxel_size))
                c = voxel_default.get(vidx, 0)
                if c < int(args_cli.max_per_voxel_front):
                    voxel_default[vidx] = c + 1
                    collected_default.append(row_kp)
                    accepted_default += 1
                    continue

        if it % 10 == 0:
            ok_count = int(ok.sum().item())
            ok_z_min = float(ee_pos_plb[ok, 2].min()) if ok_count > 0 else float("nan")
            ok_z_max = float(ee_pos_plb[ok, 2].max()) if ok_count > 0 else float("nan")
            ok_x_min = float(ee_pos_plb[ok, 0].min()) if ok_count > 0 else float("nan")
            ok_x_max = float(ee_pos_plb[ok, 0].max()) if ok_count > 0 else float("nan")

            n_default = int((posture_id == 0).sum().item())
            n_squat = int((posture_id == 1).sum().item())

            print(
                f"[INFO] iter={it} probs=({balanced_probs[0]:.2f},{balanced_probs[1]:.2f},{balanced_probs[2]:.2f}) "
                f"task={len(collected_task)}/{target_task} "
                f"mid={len(collected_mid)}/{target_mid} "
                f"default={len(collected_default)}/{target_default} "
                f"accepted=[task:{accepted_task}, mid:{accepted_mid}, default:{accepted_default}] "
                f"ok={ok_count}/{num_envs} "
                f"posture=[default:{n_default}, squat:{n_squat}] "
                f"ok_x=[{ok_x_min:.3f},{ok_x_max:.3f}] "
                f"raw_z=[{float(ee_pos_plb[:, 2].min()):.3f},{float(ee_pos_plb[:, 2].max()):.3f}] "
                f"ok_z=[{ok_z_min:.3f},{ok_z_max:.3f}] "
                f"base_z_mean={float(base_pos_w[:, 2].mean()):.3f}"
            )

    if len(collected_task) + len(collected_mid) + len(collected_default) == 0:
        print("[WARN] no samples collected.")
        sim.stop()
        return

    task = np.asarray(collected_task, dtype=np.float32)
    mid = np.asarray(collected_mid, dtype=np.float32)
    default = np.asarray(collected_default, dtype=np.float32)

    out_parts = []
    bucket_parts = []

    # workspace_id: 3 = task low-z max-squat, 2 = mid-z squat interpolation, 1 = default posture
    if task.shape[0] > 0:
        out_parts.append(task)
        bucket_parts.append(np.full((task.shape[0],), 3, dtype=np.int64))
    if mid.shape[0] > 0:
        out_parts.append(mid)
        bucket_parts.append(np.full((mid.shape[0],), 2, dtype=np.int64))
    if default.shape[0] > 0:
        out_parts.append(default)
        bucket_parts.append(np.full((default.shape[0],), 1, dtype=np.int64))

    out_all = np.concatenate(out_parts, axis=0)
    bucket_all = np.concatenate(bucket_parts, axis=0)

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
    workspace_id = bucket_all[keep]

    perm = rng.permutation(out.shape[0])
    out = out[perm]
    workspace_id = workspace_id[perm]

    out_path = args_cli.out
    out_dir = os.path.dirname(out_path)

    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    np.save(out_path, out)

    task_final = out[workspace_id == 3]
    mid_final = out[workspace_id == 2]
    default_final = out[workspace_id == 1]

    n0 = int((workspace_id == 1).sum())
    n1 = int((workspace_id != 1).sum())
    total_final = max(1, int(out.shape[0]))

    print(f"[DONE] saved: {out_path}")
    print(
        "[DONE] collected before QC: "
        f"task={len(collected_task)}, mid={len(collected_mid)}, default={len(collected_default)}"
    )
    print(f"[DONE] final shape: {out.shape}")
    print(
        "[DONE] final workspace composition: "
        f"task_low_z={task_final.shape[0]} ({task_final.shape[0] / total_final:.3f}), "
        f"mid_z={mid_final.shape[0]} ({mid_final.shape[0] / total_final:.3f}), "
        f"default={default_final.shape[0]} ({default_final.shape[0] / total_final:.3f})"
    )
    print(
        "[DONE] final posture composition: "
        f"default={n0} ({n0 / total_final:.3f}), "
        f"squat={n1} ({n1 / total_final:.3f})"
    )
    print("[STATS] kp0 x min/max/mean:", float(out[:, 0].min()), float(out[:, 0].max()), float(out[:, 0].mean()))
    print("[STATS] kp0 y min/max/mean:", float(out[:, 1].min()), float(out[:, 1].max()), float(out[:, 1].mean()))
    print("[STATS] kp0 z min/max/mean:", float(out[:, 2].min()), float(out[:, 2].max()), float(out[:, 2].mean()))
    print("[STATS] task kp0 z min/max/mean:",
          float(task_final[:, 2].min()) if task_final.shape[0] > 0 else float("nan"),
          float(task_final[:, 2].max()) if task_final.shape[0] > 0 else float("nan"),
          float(task_final[:, 2].mean()) if task_final.shape[0] > 0 else float("nan"))
    print("[STATS] mid kp0 z min/max/mean:",
          float(mid_final[:, 2].min()) if mid_final.shape[0] > 0 else float("nan"),
          float(mid_final[:, 2].max()) if mid_final.shape[0] > 0 else float("nan"),
          float(mid_final[:, 2].mean()) if mid_final.shape[0] > 0 else float("nan"))

    # Recover saved keypoint orientation statistics from kp0/kp1/kp2.
    # kp1-kp0 is local +X * kp_dx, kp2-kp0 is local +Z * kp_dz in PLB.
    x_axis = out[:, 3:6] - out[:, 0:3]
    z_axis = out[:, 6:9] - out[:, 0:3]
    x_axis = x_axis / np.maximum(np.linalg.norm(x_axis, axis=1, keepdims=True), 1.0e-9)
    z_axis = z_axis / np.maximum(np.linalg.norm(z_axis, axis=1, keepdims=True), 1.0e-9)
    saved_yaw_x = np.degrees(np.arctan2(x_axis[:, 1], x_axis[:, 0]))
    saved_pitch_x = np.degrees(
        np.arctan2(x_axis[:, 2], np.sqrt(x_axis[:, 0] ** 2 + x_axis[:, 1] ** 2) + 1.0e-9)
    )
    saved_z_vertical_abs = np.abs(z_axis[:, 2])

    task_mask = workspace_id == 3
    mid_mask = workspace_id == 2
    default_mask = workspace_id == 1

    def _print_axis_stats(name, mask):
        if not np.any(mask):
            print(f"[STATS] {name} orientation: empty")
            return
        print(
            f"[STATS] {name} yaw_x min/max/mean: "
            f"{float(saved_yaw_x[mask].min())} {float(saved_yaw_x[mask].max())} {float(saved_yaw_x[mask].mean())}"
        )
        print(
            f"[STATS] {name} pitch_x min/max/mean: "
            f"{float(saved_pitch_x[mask].min())} {float(saved_pitch_x[mask].max())} {float(saved_pitch_x[mask].mean())}"
        )
        print(
            f"[STATS] {name} |local_z_vertical| min/max/mean: "
            f"{float(saved_z_vertical_abs[mask].min())} {float(saved_z_vertical_abs[mask].max())} {float(saved_z_vertical_abs[mask].mean())}"
        )

    _print_axis_stats("all", np.ones_like(workspace_id, dtype=bool))
    _print_axis_stats("task", task_mask)
    _print_axis_stats("mid", mid_mask)
    _print_axis_stats("default", default_mask)

    print("[CHECK] mean ||kp1-kp0||:", float(np.mean(np.linalg.norm(out[:, 3:6] - out[:, 0:3], axis=1))))
    print("[CHECK] mean ||kp2-kp0||:", float(np.mean(np.linalg.norm(out[:, 6:9] - out[:, 0:3], axis=1))))

    sim.stop()


if __name__ == "__main__":
    main()
    simulation_app.close()