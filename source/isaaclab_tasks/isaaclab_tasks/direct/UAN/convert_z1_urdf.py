# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert the standalone Unitree Z1 URDF into the USD the UAN task loads.

The repo ships ``Robots/Unitree/Z1/urdf/z1.urdf`` but no Z1-only USD -- the
only Z1 in USD form is fused into the B2WZ1 asset. The UAN dataset was
collected with the arm bench-mounted, base level and world-aligned, gripper
empty, so the training asset should be exactly that: a fixed-base, 6-DOF,
gripper-less Z1. Simulating the 20 unused B2W joints instead would cost
parallel environments for nothing.

z1.urdf already declares a ``world`` link and a fixed ``base_static_joint``,
and stops at ``link06`` with no gripper, so the conversion is direct.

Joint drives are created with ZERO stiffness and damping on purpose. The UAN
environment applies the nominal PD law itself so it can add the learned
residual AFTER the torque-speed clip; if PhysX also ran a position loop the
two would fight and the applied torque would not be what the reward assumes.

Run with Isaac Sim's python:

    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/UAN/convert_z1_urdf.py

Add ``--headless`` on a machine with no display.
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert the Z1 URDF to USD for UAN training.")
parser.add_argument(
    "--urdf",
    type=str,
    default=None,
    help="Source URDF (default: the Z1 URDF shipped in isaaclab_assets).",
)
parser.add_argument(
    "--usd",
    type=str,
    default=None,
    help="Destination USD (default: Robots/Unitree/Z1/usd/z1.usd, which is "
    "what UNITREE_Z1_UAN_CFG points at).",
)
parser.add_argument(
    "--force",
    action="store_true",
    help="Overwrite an existing USD instead of leaving it alone.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR  # noqa: E402

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


def main() -> None:
    z1_dir = os.path.join(ISAACLAB_ASSETS_DATA_DIR, "Robots", "Unitree", "Z1")
    urdf_path = args_cli.urdf or os.path.join(z1_dir, "urdf", "z1.urdf")
    usd_path = args_cli.usd or os.path.join(z1_dir, "usd", "z1.usd")

    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    if os.path.isfile(usd_path) and not args_cli.force:
        print(f"[UAN] {usd_path} already exists; pass --force to overwrite.")
        return

    os.makedirs(os.path.dirname(usd_path), exist_ok=True)

    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=os.path.dirname(usd_path),
        usd_file_name=os.path.basename(usd_path),
        # The URDF's own `world` link plus `base_static_joint` already pin the
        # arm; this makes the root fixed at the articulation level too, which
        # is what the bench mount was.
        fix_base=True,
        merge_fixed_joints=True,
        convert_mimic_joints_to_normal_joints=False,
        self_collision=False,
        # Meshes are simple; a convex hull per link is plenty and much
        # cheaper than decomposition for an arm that never contacts anything
        # during UAN training.
        collider_type="convex_hull",
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="none",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness={j: 0.0 for j in ARM_JOINTS},
                damping={j: 0.0 for j in ARM_JOINTS},
            ),
        ),
        force_usd_conversion=args_cli.force,
    )

    print("[UAN] converting")
    print(f"[UAN]   from {urdf_path}")
    print(f"[UAN]   to   {usd_path}")
    converter = UrdfConverter(cfg)
    print(f"[UAN] done: {converter.usd_path}")
    print(
        "[UAN] joint drives are zero-gain by design -- the UAN environment "
        "applies the nominal PD law itself."
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        # Always release the app, but never swallow the traceback: a silent
        # failure here leaves a missing or half-written USD that only shows
        # up much later as an unhelpful load error.
        simulation_app.close()
