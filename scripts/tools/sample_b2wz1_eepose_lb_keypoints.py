"""
Sample collision-free reachable EE poses (pos+quat) in level-base frame (LB) for Unitree B2W+Z1,
then convert to 3 keypoints and save as .npy.

LB frame definition:
- Origin: base frame origin (x,y,z)
- Orientation: keep base yaw, roll/pitch set to 0

Output per sample (in LB): (N, 9)
[kp0(3), kp1(3), kp2(3)]
where:
kp0 = ee_pos_lb
kp1 = kp0 + R(ee_quat_lb) * [dx, 0, 0]
kp2 = kp0 + R(ee_quat_lb) * [0, 0, dz]

Hard constraints / filters (task-relevant):
- EE +X axis yaw in [-90, +90] deg
- EE +X axis pitch in [-60, +60] deg
- EE +Z axis must not be flipped: z_ee_lb.z > 0
- EE position must be in front: kp0_x > front_min_x

Quality checks before saving:
- Orthogonality: (kp1-kp0) · (kp2-kp0) near 0
- Norm checks: ||kp1-kp0|| ~ dx, ||kp2-kp0|| ~ dz

Collision-free:
- ContactSensor on arm links, threshold on net forces

Usage:
./isaaclab.sh -p scripts/tools/sample_b2wz1_eepose_lb_keypoints.py --out scripts/tools/reachable_kp0kp1kp2_lb.npy
"""

import argparse
from isaaclab.app import AppLauncher

# Simulation parameters and file paths
parser = argparse.ArgumentParser(description="Sample collision-free reachable EE pose in LB, then save keypoints.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments to spawn.")
parser.add_argument("--num_samples", type=int, default=10000, help="How many valid samples to collect.")
parser.add_argument("--out", type=str, default="scripts/tools/reachable_kp0kp1kp2_lb.npy", help="Output npy file path.")

# Collision check parameters
parser.add_argument("--settle_steps", type=int, default=60, help="Physics steps after reset before checking collision.")
parser.add_argument("--force_threshold", type=float, default=3.0, help="Contact force threshold (N) for collision.")
parser.add_argument("--arm_joints", type=str, default="joint1,joint2,joint3,joint4,joint5,joint6", help="Comma-separated arm joint names.")

# Joint-limit filter
parser.add_argument("--joint_margin_deg", type=float, default=20.0, help="Filter out samples too close to arm joint limits.")

# Workspace uniformization
parser.add_argument("--voxel_size", type=float, default=0.03, help="Workspace voxel size (m) in LB frame.")
parser.add_argument("--max_per_voxel", type=int, default=5, help="Max samples per voxel cell (by kp0 position).")

# Hard position constraint
parser.add_argument("--front_min_x", type=float, default=0.00, help="Hard constraint: kp0_x must be > front_min_x (m).")

# Hard orientation constraints (camera cone)
parser.add_argument("--yaw_min_deg", type=float, default=-90.0, help="Hard constraint: yaw(x_ee) >= yaw_min_deg.")
parser.add_argument("--yaw_max_deg", type=float, default=90.0, help="Hard constraint: yaw(x_ee) <= yaw_max_deg.")
parser.add_argument("--pitch_min_deg", type=float, default=-60.0, help="Hard constraint: pitch(x_ee) >= pitch_min_deg.")
parser.add_argument("--pitch_max_deg", type=float, default=60.0, help="Hard constraint: pitch(x_ee) <= pitch_max_deg.")
parser.add_argument("--require_z_up", action="store_true", help="If set: require z_ee_lb.z > 0 (no flip).")
parser.set_defaults(require_z_up=True)

# SOFT forward bias
parser.add_argument("--x_bias", type=float, default=0.40, help="Position forward bias strength in [0,1]. 0=off.")
parser.add_argument("--x0", type=float, default=0.30, help="Sigmoid center for forward position preference (m).")
parser.add_argument("--x_sigma", type=float, default=0.12, help="Sigmoid smoothness (m). Smaller=sharper.")
parser.add_argument("--x_floor", type=float, default=0.20, help="Minimum acceptance prob floor for position bias.")

# Keypoint offsets (EE local frame)
parser.add_argument("--kp_dx", type=float, default=0.30, help="Offset along EE +X for kp1 (m).")
parser.add_argument("--kp_dz", type=float, default=0.30, help="Offset along EE +Z for kp2 (m).")

# Quality checks
parser.add_argument("--ortho_tol", type=float, default=2e-3, help="Max |cos(theta)| between (kp1-kp0) and (kp2-kp0).")
parser.add_argument("--norm_tol", type=float, default=3e-3, help="Max abs error in ||kp1-kp0|| vs dx and ||kp2-kp0|| vs dz.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import os
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import UNITREE_B2WZ1_CFG  # isort: skip

ARM_BODIES_REGEX = "(link01|link02|link03|link04|link05|link06|gripperStator|gripperMover)"

# Z1 joint limits (deg)
Z1_LIMITS_DEG = {
    "joint1": (-150.0, 150.0),
    "joint2": (0.0, 180.0),
    "joint3": (-165.0, 0.0),
    "joint4": (-80.0, 80.0),
    "joint5": (-85.0, 85.0),
    "joint6": (-160.0, 160.0),
}


def _deg2rad(x: float) -> float:
    return float(x) * np.pi / 180.0


@configclass
class SampleSceneCfg(InteractiveSceneCfg):
    """Scene: plane + light + Unitree B2WZ1 + contact sensor on arm links."""
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # Fix root for stable sampling (no base bouncing)
    robot: ArticulationCfg = UNITREE_B2WZ1_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UNITREE_B2WZ1_CFG.spawn.replace(
            articulation_props=UNITREE_B2WZ1_CFG.spawn.articulation_props.replace(
                fix_root_link=True
            )
        ),
    )

    contact_forces_arm = ContactSensorCfg(
        prim_path=(f"{{ENV_REGEX_NS}}/Robot/{ARM_BODIES_REGEX}"),
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )


# Frame helper - LB frame: origin at base, yaw-only (ignore roll/pitch)
def yaw_only_quat(q_wxyz_batched: torch.Tensor) -> torch.Tensor:
    """Extract yaw-only quaternion from full quaternion (wxyz), batched (N,4)."""
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(q_wxyz_batched)
    zeros = torch.zeros_like(yaw)
    return math_utils.quat_from_euler_xyz(zeros, zeros, yaw)


def _resolve_ids(robot, arm_joint_names, ee_body_name: str):
    if not hasattr(robot, "find_joints"):
        raise RuntimeError("Robot articulation does not expose find_joints(). Please check IsaacLab version/API.")
    arm_joint_ids, _ = robot.find_joints(arm_joint_names)

    if not hasattr(robot, "find_bodies"):
        raise RuntimeError("Robot articulation does not expose find_bodies(). Please check IsaacLab version/API.")
    ee_body_ids, _ = robot.find_bodies(ee_body_name)
    ee_body_id = int(ee_body_ids[0])
    return arm_joint_ids, ee_body_id


def _get_joint_limits_rad(robot, arm_joint_names, arm_joint_ids: torch.Tensor):
    lim = robot.data.soft_joint_pos_limits  # (N, num_joints, 2)
    low = lim[0, arm_joint_ids, 0].clone()
    high = lim[0, arm_joint_ids, 1].clone()

    for i, jn in enumerate(arm_joint_names):
        if jn in Z1_LIMITS_DEG:
            lo_deg, hi_deg = Z1_LIMITS_DEG[jn]
            low[i] = _deg2rad(lo_deg)
            high[i] = _deg2rad(hi_deg)
    return low, high


def _sample_arm_q(robot, arm_joint_names, arm_joint_ids: torch.Tensor):
    """Mixture sampling: 80% uniform + 20% soft forward bias."""
    low, high = _get_joint_limits_rad(robot, arm_joint_names, arm_joint_ids)
    num_envs = robot.data.joint_pos.shape[0]
    J = arm_joint_ids.numel()

    u = torch.rand((num_envs, J), device=robot.device)
    q_uni = low.unsqueeze(0) + (high - low).unsqueeze(0) * u

    q_ref_deg = torch.tensor([0.0, 120.0, -120.0, 0.0, 0.0, 0.0], device=robot.device)
    q_ref = q_ref_deg * (np.pi / 180.0)

    noise_std_deg = torch.tensor([25.0, 18.0, 18.0, 30.0, 30.0, 35.0], device=robot.device)
    noise_std = noise_std_deg * (np.pi / 180.0)

    q_biased = q_ref.unsqueeze(0) + torch.randn((num_envs, J), device=robot.device) * noise_std.unsqueeze(0)
    q_biased = torch.max(torch.min(q_biased, high.unsqueeze(0)), low.unsqueeze(0))

    mix = (torch.rand((num_envs, 1), device=robot.device) < 0.20).expand(-1, J)
    return torch.where(mix, q_biased, q_uni)


def _ok_not_near_joint_limits(
    robot, joint_pos: torch.Tensor, arm_joint_names, arm_joint_ids: torch.Tensor, margin_deg: float
) -> torch.Tensor:
    margin = _deg2rad(margin_deg)
    low, high = _get_joint_limits_rad(robot, arm_joint_names, arm_joint_ids)
    q = joint_pos[:, arm_joint_ids]
    min_margin = torch.minimum(q - low.unsqueeze(0), high.unsqueeze(0) - q).min(dim=1).values
    return min_margin > margin


def _reset_all_envs(scene: InteractiveScene, arm_joint_ids: torch.Tensor, q_arm: torch.Tensor):
    """For fixed-root robot, only need to reset joints."""
    robot = scene["robot"]

    # Reset first to avoid overwriting our writes
    scene.reset()

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, arm_joint_ids] = q_arm
    joint_vel[:, arm_joint_ids] = 0.0
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    return joint_pos


def _voxel_index(pos_lb_np: np.ndarray, voxel_size: float):
    return tuple(np.floor(pos_lb_np / voxel_size).astype(np.int32).tolist())


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = np.exp(-z)
        return float(1.0 / (1.0 + ez))
    ez = np.exp(z)
    return float(ez / (1.0 + ez))


def _accept_prob_position_soft(x: float, x_bias: float, x0: float, x_sigma: float, x_floor: float) -> float:
    """
    p_pos = (1-x_bias) + x_bias * (x_floor + (1-x_floor)*sigmoid((x-x0)/x_sigma))
    """
    x_bias = float(np.clip(x_bias, 0.0, 1.0))
    x_floor = float(np.clip(x_floor, 0.0, 1.0))
    z = (x - x0) / max(1e-6, x_sigma)
    s = _sigmoid(z)
    shaped = x_floor + (1.0 - x_floor) * s
    p = (1.0 - x_bias) + x_bias * shaped
    return float(np.clip(p, 1e-6, 1.0))


def _orthogonality_ok(
    kp0: np.ndarray,
    kp1: np.ndarray,
    kp2: np.ndarray,
    ortho_tol: float,
    dx: float,
    dz: float,
    norm_tol: float,
):
    v1 = kp1 - kp0
    v2 = kp2 - kp0
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return False, {"reason": "zero_norm"}
    if abs(n1 - dx) > norm_tol or abs(n2 - dz) > norm_tol:
        return False, {"reason": "norm_mismatch", "n1": n1, "n2": n2}
    cos = float(np.dot(v1, v2) / (n1 * n2))
    if abs(cos) > ortho_tol:
        return False, {"reason": "not_orthogonal", "cos": cos}
    return True, {"cos": cos, "n1": n1, "n2": n2}


@torch.no_grad()
def main():
    device = args_cli.device
    arm_joint_names = [s.strip() for s in args_cli.arm_joints.split(",")]
    ee_body_name = "gripperStator"

    collision_threshold = float(args_cli.force_threshold)
    joint_margin_deg = float(args_cli.joint_margin_deg)
    voxel_size = float(args_cli.voxel_size)
    max_per_voxel = int(args_cli.max_per_voxel)

    front_min_x = float(args_cli.front_min_x)

    yaw_min = float(args_cli.yaw_min_deg)
    yaw_max = float(args_cli.yaw_max_deg)
    pitch_min = float(args_cli.pitch_min_deg)
    pitch_max = float(args_cli.pitch_max_deg)
    require_z_up = bool(args_cli.require_z_up)

    x_bias = float(args_cli.x_bias)
    x0 = float(args_cli.x0)
    x_sigma = float(args_cli.x_sigma)
    x_floor = float(args_cli.x_floor)

    dx = float(args_cli.kp_dx)
    dz = float(args_cli.kp_dz)

    ortho_tol = float(args_cli.ortho_tol)
    norm_tol = float(args_cli.norm_tol)

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
    arm_joint_ids, ee_body_id = _resolve_ids(robot, arm_joint_names, ee_body_name)
    if not isinstance(arm_joint_ids, torch.Tensor):
        arm_joint_ids = torch.tensor(arm_joint_ids, device=robot.device, dtype=torch.long)

    print("[INFO] Arm joints:", arm_joint_names)
    print("[INFO] Arm joint ids:", arm_joint_ids.detach().cpu().tolist())
    print("[INFO] EE body:", ee_body_name, "body_id:", ee_body_id)
    print("[INFO] Contact sensor prim regex:", scene_cfg.contact_forces_arm.prim_path)
    print(f"[INFO] settle_steps={int(args_cli.settle_steps)}, force_threshold={collision_threshold}N")
    print(f"[INFO] Joint margin filter: {joint_margin_deg} deg")
    print(f"[INFO] Workspace uniformization: voxel_size={voxel_size} m, max_per_voxel={max_per_voxel}")
    print(f"[INFO] Hard pos: kp0_x > {front_min_x:.3f}")
    print(f"[INFO] Hard ori: yaw(x_ee)∈[{yaw_min},{yaw_max}] deg, pitch(x_ee)∈[{pitch_min},{pitch_max}] deg, z_ee.z>0={require_z_up}")
    print(f"[INFO] Soft forward bias: x_bias={x_bias}, x0={x0}, x_sigma={x_sigma}, x_floor={x_floor}")
    print(f"[INFO] Keypoints offsets: dx={dx}, dz={dz}")
    print(f"[INFO] Quality: ortho_tol={ortho_tol}, norm_tol={norm_tol}")

    target = int(args_cli.num_samples)
    num_envs = robot.data.joint_pos.shape[0]

    voxel_counts: dict[tuple, int] = {}
    rng = np.random.default_rng()

    collected_pose_lb = []  # (N,7): [px,py,pz,qw,qx,qy,qz] in LB

    ex = torch.tensor([1.0, 0.0, 0.0], device=robot.device, dtype=torch.float32)
    ez = torch.tensor([0.0, 0.0, 1.0], device=robot.device, dtype=torch.float32)

    deg = 180.0 / np.pi

    it = 0
    while len(collected_pose_lb) < target and simulation_app.is_running():
        it += 1

        q_arm = _sample_arm_q(robot, arm_joint_names, arm_joint_ids)
        joint_pos = _reset_all_envs(scene, arm_joint_ids, q_arm)

        # settle
        for _ in range(int(args_cli.settle_steps)):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

        # collision filter
        contact = scene["contact_forces_arm"]
        forces_w = contact.data.net_forces_w
        force_norm = torch.linalg.norm(forces_w, dim=-1)
        collided = (force_norm > collision_threshold).any(dim=1)
        ok = ~collided

        # joint margin filter
        ok = ok & _ok_not_near_joint_limits(robot, joint_pos, arm_joint_names, arm_joint_ids, joint_margin_deg)

        if int(ok.sum().item()) == 0:
            if it % 10 == 0:
                print(f"[INFO] iter={it}: 0 ok envs after collision+joint filters.")
            continue

        # --- EE pose in LB ---
        base_pos_w = robot.data.root_pos_w
        base_quat_w = robot.data.root_quat_w

        lb_pos_w = base_pos_w.clone()                      # origin at base origin
        lb_quat_w = yaw_only_quat(base_quat_w)             # yaw-only
        lb_quat_inv = math_utils.quat_conjugate(lb_quat_w)

        ee_pos_w = robot.data.body_pos_w[:, ee_body_id, :]
        ee_quat_w = robot.data.body_quat_w[:, ee_body_id, :]

        ee_pos_lb = math_utils.quat_apply_inverse(lb_quat_w, ee_pos_w - lb_pos_w)
        ee_quat_lb = math_utils.quat_mul(lb_quat_inv, ee_quat_w)
        ee_quat_lb = math_utils.quat_unique(ee_quat_lb)

        # axes in LB
        x_ee_lb = math_utils.quat_apply(ee_quat_lb, ex.unsqueeze(0).expand_as(ee_pos_lb))
        z_ee_lb = math_utils.quat_apply(ee_quat_lb, ez.unsqueeze(0).expand_as(ee_pos_lb))

        # ---- Hard orientation constraints ----
        # yaw and pitch of x_ee_lb
        yaw = torch.atan2(x_ee_lb[:, 1], x_ee_lb[:, 0])
        pitch = torch.atan2(x_ee_lb[:, 2], torch.sqrt(x_ee_lb[:, 0] ** 2 + x_ee_lb[:, 1] ** 2) + 1e-9)
        yaw_deg = yaw * deg
        pitch_deg = pitch * deg

        ok = ok & (yaw_deg >= yaw_min) & (yaw_deg <= yaw_max)
        ok = ok & (pitch_deg >= pitch_min) & (pitch_deg <= pitch_max)

        if require_z_up:
            ok = ok & (z_ee_lb[:, 2] > 0.0)

        # ---- Hard position constraint ----
        ok = ok & (ee_pos_lb[:, 0] > front_min_x)

        num_ok = int(ok.sum().item())
        if num_ok == 0:
            if it % 10 == 0:
                print(f"[INFO] iter={it}: 0 ok envs after hard ori+pos constraints.")
            continue

        ee_pose_lb = torch.cat([ee_pos_lb, ee_quat_lb], dim=-1)  # (N,7)

        pose_ok = ee_pose_lb[ok].detach().cpu().numpy()
        pos_ok = ee_pos_lb[ok].detach().cpu().numpy()

        accepted = 0
        rej_posbias = 0
        rej_voxel = 0

        for row_pose, row_pos in zip(pose_ok, pos_ok):
            # soft forward bias
            p_pos = _accept_prob_position_soft(
                float(row_pos[0]), x_bias=x_bias, x0=x0, x_sigma=x_sigma, x_floor=x_floor
            )
            if rng.random() > p_pos:
                rej_posbias += 1
                continue

            # voxel uniformization by kp0 position
            vidx = _voxel_index(row_pos, voxel_size)
            c = voxel_counts.get(vidx, 0)
            if c >= max_per_voxel:
                rej_voxel += 1
                continue

            voxel_counts[vidx] = c + 1
            collected_pose_lb.append(row_pose)
            accepted += 1
            if len(collected_pose_lb) >= target:
                break

        if it % 10 == 0 or len(collected_pose_lb) >= target:
            print(
                f"[INFO] iter={it}: ok={num_ok}/{num_envs}, accepted={accepted}, "
                f"collected_pose={len(collected_pose_lb)}/{target}, voxels={len(voxel_counts)}, "
                f"rej_posbias={rej_posbias}, rej_voxel={rej_voxel}"
            )

    poses = np.asarray(collected_pose_lb, dtype=np.float32)
    if poses.shape[0] == 0:
        print("[WARN] No samples collected. Check thresholds / constraints.")
        sim.stop()
        return

    # --- Convert to keypoints ---
    poses_t = torch.from_numpy(poses).to(device=robot.device, dtype=torch.float32)
    p = poses_t[:, 0:3]
    q = poses_t[:, 3:7]

    kp0 = p
    off_x = torch.tensor([dx, 0.0, 0.0], device=robot.device, dtype=torch.float32).unsqueeze(0).expand_as(p)
    off_z = torch.tensor([0.0, 0.0, dz], device=robot.device, dtype=torch.float32).unsqueeze(0).expand_as(p)

    kp1 = kp0 + math_utils.quat_apply(q, off_x)
    kp2 = kp0 + math_utils.quat_apply(q, off_z)

    out_all = torch.cat([kp0, kp1, kp2], dim=-1).detach().cpu().numpy().astype(np.float32)

    # --- Quality checks ---
    keep = []
    rej_ortho = 0
    rej_norm = 0
    for i in range(out_all.shape[0]):
        k0 = out_all[i, 0:3]
        k1 = out_all[i, 3:6]
        k2 = out_all[i, 6:9]
        ok_q, info = _orthogonality_ok(k0, k1, k2, ortho_tol=ortho_tol, dx=dx, dz=dz, norm_tol=norm_tol)
        if not ok_q:
            if info.get("reason") == "not_orthogonal":
                rej_ortho += 1
            elif info.get("reason") == "norm_mismatch":
                rej_norm += 1
            continue
        keep.append(i)

    out = out_all[np.asarray(keep, dtype=np.int64)]
    if out.shape[0] == 0:
        print("[WARN] All samples rejected by quality checks. Relax ortho_tol/norm_tol.")
        sim.stop()
        return

    out_path = args_cli.out
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    np.save(out_path, out)

    print(f"[DONE] Saved {out.shape[0]} / {out_all.shape[0]} samples to: {out_path}")
    print(f"[DONE] shape: {out.shape} (N,9) [kp0(3), kp1(3), kp2(3)] in LB")
    print(f"[DONE] quality rejects: ortho={rej_ortho}, norm={rej_norm}")
    print("[CHECK] mean ||kp1-kp0|| =", float(np.mean(np.linalg.norm(out[:, 3:6] - out[:, 0:3], axis=1))))
    print("[CHECK] mean ||kp2-kp0|| =", float(np.mean(np.linalg.norm(out[:, 6:9] - out[:, 0:3], axis=1))))

    sim.stop()


if __name__ == "__main__":
    main()
    simulation_app.close()
