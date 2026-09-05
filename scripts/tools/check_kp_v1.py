"""
Offline check for keypoints-table distribution.

Input: .npy with shape (N,9): [kp0(3), kp1(3), kp2(3)]
Assumes:
  v1 = kp1 - kp0  (EE +X direction * dx)
  v2 = kp2 - kp0  (EE +Z direction * dz)

Outputs:
- Sanity checks: ||v1||, ||v2||, orthogonality
- kp0 workspace statistics:
    * per-axis stats for kp0_x, kp0_y, kp0_z
    * absolute min/max box
    * percentile box (default 1% ~ 99%)
    * radial distance stats in XY and 3D
- Orientation distributions:
    * yaw of v1 in XY plane: atan2(v1_y, v1_x)  (deg)
    * pitch of v1 (elevation): atan2(v1_z, sqrt(v1_x^2+v1_y^2)) (deg)
    * roll about EE local +X, reconstructed using the same zero-roll convention
      as modules/command_adapter.py (deg)
    * tilt of v2 from +Z: acos(v2_z/||v2||) (deg)
    * absolute and percentile yaw/pitch/roll ranges
- Coverage stats: percent with v1_x>0, v2_z>0, etc.
- Optional plots (matplotlib) if --plot is set.

Usage:
  python scripts/tools/check_kp.py --file scripts/tools/reachable_kp0kp1kp2_lb.npy --plot
"""

import argparse
import numpy as np


def _wrap_pi(a: np.ndarray) -> np.ndarray:
    """Wrap angle (rad) to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def _percent(x: np.ndarray) -> float:
    return float(100.0 * np.mean(x))


def _hist_summary_deg(angles_deg: np.ndarray, bins: int = 36):
    hist, edges = np.histogram(angles_deg, bins=bins, range=(-180.0, 180.0))
    return hist, edges


def _print_top_hist_bins(name: str, angles_deg: np.ndarray, bins: int):
    hist, edges = _hist_summary_deg(angles_deg, bins=bins)
    occupied = int(np.sum(hist > 0))
    print(
        f"\n{name} histogram bins: {bins}, occupied bins: "
        f"{occupied}/{bins} ({100.0 * occupied / bins:.1f}%)"
    )
    top = np.argsort(hist)[::-1][:10]
    print(f"Top {name.lower()} bins (count, range_deg):")
    for i in top:
        if hist[i] == 0:
            break
        lo, hi = edges[i], edges[i + 1]
        print(f"  {hist[i]:6d}  [{lo:7.1f}, {hi:7.1f})")


def _print_stats(name: str, x: np.ndarray):
    qs = np.percentile(x, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    print(
        f"{name}: mean={x.mean():.4f}, std={x.std():.4f}, "
        f"min={qs[0]:.4f}, p1={qs[1]:.4f}, p5={qs[2]:.4f}, "
        f"p25={qs[3]:.4f}, p50={qs[4]:.4f}, p75={qs[5]:.4f}, "
        f"p95={qs[6]:.4f}, p99={qs[7]:.4f}, max={qs[8]:.4f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, required=True, help="Path to kp table .npy (N,9).")
    ap.add_argument("--dx", type=float, default=0.30, help="Expected ||kp1-kp0||.")
    ap.add_argument("--dz", type=float, default=0.30, help="Expected ||kp2-kp0||.")
    ap.add_argument("--tol", type=float, default=5e-3, help="Tolerance for norm sanity checks.")
    ap.add_argument("--bins", type=int, default=36, help="Histogram bins for yaw/angles.")
    ap.add_argument(
        "--workspace-percentile",
        type=float,
        nargs=2,
        default=(1.0, 99.0),
        metavar=("LOW", "HIGH"),
        help="Percentile range used to summarize kp0 workspace, e.g. 1 99.",
    )
    ap.add_argument(
        "--orientation-percentile",
        type=float,
        nargs=2,
        default=(1.0, 99.0),
        metavar=("LOW", "HIGH"),
        help="Percentile range used to summarize yaw/pitch/roll, e.g. 1 99.",
    )
    ap.add_argument("--plot", action="store_true", help="Show matplotlib plots.")
    args = ap.parse_args()

    arr = np.load(args.file).astype(np.float64)
    if arr.ndim != 2 or arr.shape[1] != 9:
        raise ValueError(f"Expected (N,9), got {arr.shape}")

    p_lo, p_hi = args.workspace_percentile
    if not (0.0 <= p_lo < p_hi <= 100.0):
        raise ValueError(
            f"--workspace-percentile must satisfy 0 <= low < high <= 100, got {args.workspace_percentile}"
        )

    ori_p_lo, ori_p_hi = args.orientation_percentile
    if not (0.0 <= ori_p_lo < ori_p_hi <= 100.0):
        raise ValueError(
            "--orientation-percentile must satisfy 0 <= low < high <= 100, "
            f"got {args.orientation_percentile}"
        )

    kp0 = arr[:, 0:3]
    kp1 = arr[:, 3:6]
    kp2 = arr[:, 6:9]

    v1 = kp1 - kp0
    v2 = kp2 - kp0

    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)

    # Normalize directions
    v1u = v1 / np.clip(n1[:, None], 1e-12, None)
    v2u = v2 / np.clip(n2[:, None], 1e-12, None)

    # Orthogonality cosine
    cos_12 = np.sum(v1u * v2u, axis=1)

    # kp0 stats
    kp0_x = kp0[:, 0]
    kp0_y = kp0[:, 1]
    kp0_z = kp0[:, 2]
    kp0_r_xy = np.linalg.norm(kp0[:, :2], axis=1)
    kp0_r_3d = np.linalg.norm(kp0, axis=1)

    # v1 yaw in XY (rad->deg)
    yaw = np.arctan2(v1u[:, 1], v1u[:, 0])
    yaw = _wrap_pi(yaw)
    yaw_deg = np.degrees(yaw)

    # v1 elevation (pitch) relative to XY plane
    xy = np.sqrt(v1u[:, 0] ** 2 + v1u[:, 1] ** 2)
    elev = np.arctan2(v1u[:, 2], xy)
    elev_deg = np.degrees(elev)

    # Reconstruct roll using the same convention as modules/command_adapter.py:
    #   - v1u is EE local +X.
    #   - zero-roll local +Z is the most-upward unit vector orthogonal to +X.
    #   - positive roll follows the right-hand rule around EE local +X:
    #       z(roll) = cos(roll) * z(0) + sin(roll) * cross(x, z(0))
    up = np.zeros_like(v1u)
    up[:, 2] = 1.0
    v2_zero = up - np.sum(up * v1u, axis=1, keepdims=True) * v1u
    v2_zero_norm = np.linalg.norm(v2_zero, axis=1, keepdims=True)

    # Exact match to the adapter's singularity fallback when local +X is near +/-Z.
    fallback = np.stack(
        [-np.sin(yaw), np.cos(yaw), np.zeros_like(yaw)],
        axis=-1,
    )
    fallback /= np.clip(np.linalg.norm(fallback, axis=1, keepdims=True), 1e-12, None)
    v2_zero = np.where(
        v2_zero_norm > 1.0e-6,
        v2_zero / np.clip(v2_zero_norm, 1e-12, None),
        fallback,
    )

    roll_tangent = np.cross(v1u, v2_zero)
    cos_roll = np.sum(v2u * v2_zero, axis=1)
    sin_roll = np.sum(v2u * roll_tangent, axis=1)
    roll = _wrap_pi(np.arctan2(sin_roll, cos_roll))
    roll_deg = np.degrees(roll)

    # v2 tilt from +Z
    v2z = np.clip(v2u[:, 2], -1.0, 1.0)
    tilt = np.arccos(v2z)  # 0 means perfectly up
    tilt_deg = np.degrees(tilt)

    N = arr.shape[0]
    print(f"\n[FILE] {args.file}")
    print(f"[N] {N}")

    print("\n--- Sanity checks ---")
    print(
        f"||v1|| mean={n1.mean():.6f}, std={n1.std():.6f}, "
        f"min={n1.min():.6f}, max={n1.max():.6f}  (expect ~{args.dx})"
    )
    print(
        f"||v2|| mean={n2.mean():.6f}, std={n2.std():.6f}, "
        f"min={n2.min():.6f}, max={n2.max():.6f}  (expect ~{args.dz})"
    )
    print(
        f"cos(v1,v2) mean={cos_12.mean():.3e}, std={cos_12.std():.3e}, "
        f"max_abs={np.max(np.abs(cos_12)):.3e}"
    )

    ok_n1 = np.abs(n1 - args.dx) < args.tol
    ok_n2 = np.abs(n2 - args.dz) < args.tol
    ok_ortho = np.abs(cos_12) < 5e-3
    print(f"pass ||v1|| within tol: {_percent(ok_n1):.2f}%")
    print(f"pass ||v2|| within tol: {_percent(ok_n2):.2f}%")
    print(f"pass orthogonality |cos|<5e-3: {_percent(ok_ortho):.2f}%")

    print("\n--- KP0 workspace statistics ---")
    _print_stats("kp0_x", kp0_x)
    _print_stats("kp0_y", kp0_y)
    _print_stats("kp0_z", kp0_z)
    _print_stats("kp0_r_xy", kp0_r_xy)
    _print_stats("kp0_r_3d", kp0_r_3d)

    kp0_abs_min = kp0.min(axis=0)
    kp0_abs_max = kp0.max(axis=0)
    kp0_pct_min = np.percentile(kp0, p_lo, axis=0)
    kp0_pct_max = np.percentile(kp0, p_hi, axis=0)

    print("\nKP0 absolute workspace box:")
    print(
        f"  x in [{kp0_abs_min[0]:.4f}, {kp0_abs_max[0]:.4f}], "
        f"y in [{kp0_abs_min[1]:.4f}, {kp0_abs_max[1]:.4f}], "
        f"z in [{kp0_abs_min[2]:.4f}, {kp0_abs_max[2]:.4f}]"
    )

    print(f"\nKP0 percentile workspace box ({p_lo:.1f}% ~ {p_hi:.1f}%):")
    print(
        f"  x in [{kp0_pct_min[0]:.4f}, {kp0_pct_max[0]:.4f}], "
        f"y in [{kp0_pct_min[1]:.4f}, {kp0_pct_max[1]:.4f}], "
        f"z in [{kp0_pct_min[2]:.4f}, {kp0_pct_max[2]:.4f}]"
    )

    inside_pct_box = (
        (kp0_x >= kp0_pct_min[0]) & (kp0_x <= kp0_pct_max[0]) &
        (kp0_y >= kp0_pct_min[1]) & (kp0_y <= kp0_pct_max[1]) &
        (kp0_z >= kp0_pct_min[2]) & (kp0_z <= kp0_pct_max[2])
    )
    print(f"Samples inside percentile workspace box: {_percent(inside_pct_box):.2f}%")

    print("\n--- Coverage / constraints style stats ---")
    print(f"kp0_x > 0 (workspace in front of base): {_percent(kp0_x > 0):.2f}%")
    print(f"v1_x > 0 (EE +X in front half-space): {_percent(v1u[:, 0] > 0):.2f}%")
    print(f"v2_z > 0 (EE +Z points upward-ish): {_percent(v2u[:, 2] > 0):.2f}%")
    print(f"tilt < 30 deg (Z within 30deg of up): {_percent(tilt_deg < 30.0):.2f}%")
    print(f"tilt < 60 deg: {_percent(tilt_deg < 60.0):.2f}%")

    print("\n--- Orientation distributions ---")
    _print_stats("yaw_deg(v1 in XY)", yaw_deg)
    _print_stats("pitch_deg(v1 elevation)", elev_deg)
    _print_stats("roll_deg(about EE local +X)", roll_deg)
    _print_stats("tilt_deg(v2 from +Z)", tilt_deg)

    ori_names = ["yaw", "pitch", "roll"]
    ori_values = [yaw_deg, elev_deg, roll_deg]
    ori_abs_min = np.array([x.min() for x in ori_values])
    ori_abs_max = np.array([x.max() for x in ori_values])
    ori_pct_min = np.array([np.percentile(x, ori_p_lo) for x in ori_values])
    ori_pct_max = np.array([np.percentile(x, ori_p_hi) for x in ori_values])

    print("\nOrientation absolute ranges (degrees):")
    for name, lo, hi in zip(ori_names, ori_abs_min, ori_abs_max):
        print(f"  {name:5s} in [{lo:.4f}, {hi:.4f}]")

    print(
        f"\nOrientation percentile ranges ({ori_p_lo:.1f}% ~ {ori_p_hi:.1f}%, degrees):"
    )
    for name, lo, hi in zip(ori_names, ori_pct_min, ori_pct_max):
        print(f"  {name:5s} in [{lo:.4f}, {hi:.4f}]")

    print("\nCopy-ready ENV_PARAMS ranges (percentile, radians):")
    print(
        f'  "ee_yaw_range": [math.radians({ori_pct_min[0]:.4f}), '
        f'math.radians({ori_pct_max[0]:.4f})],'
    )
    print(
        f'  "ee_pitch_range": [math.radians({ori_pct_min[1]:.4f}), '
        f'math.radians({ori_pct_max[1]:.4f})],'
    )
    print(
        f'  "ee_roll_range": [math.radians({ori_pct_min[2]:.4f}), '
        f'math.radians({ori_pct_max[2]:.4f})],'
    )

    _print_top_hist_bins("Yaw", yaw_deg, bins=args.bins)
    _print_top_hist_bins("Pitch", elev_deg, bins=args.bins)
    _print_top_hist_bins("Roll", roll_deg, bins=args.bins)

    tilt_hist, tilt_edges = np.histogram(tilt_deg, bins=36, range=(0.0, 180.0))
    occ2 = np.sum(tilt_hist > 0)
    print(f"\nTilt histogram bins: 36 (0..180), occupied: {occ2}/36 ({100.0 * occ2 / 36:.1f}%)")
    top2 = np.argsort(tilt_hist)[::-1][:10]
    print("Top tilt bins (count, range_deg):")
    for i in top2:
        if tilt_hist[i] == 0:
            break
        lo, hi = tilt_edges[i], tilt_edges[i + 1]
        print(f"  {tilt_hist[i]:6d}  [{lo:7.1f}, {hi:7.1f})")

    if args.plot:
        import matplotlib.pyplot as plt

        # kp0 axis histograms
        plt.figure()
        plt.hist(kp0_x, bins=50)
        plt.xlabel("kp0_x")
        plt.ylabel("count")
        plt.title("KP0 X distribution")
        plt.show()

        plt.figure()
        plt.hist(kp0_y, bins=50)
        plt.xlabel("kp0_y")
        plt.ylabel("count")
        plt.title("KP0 Y distribution")
        plt.show()

        plt.figure()
        plt.hist(kp0_z, bins=50)
        plt.xlabel("kp0_z")
        plt.ylabel("count")
        plt.title("KP0 Z distribution")
        plt.show()

        # kp0 XY scatter
        M = min(N, 20000)
        idx = np.random.default_rng(0).choice(N, size=M, replace=False) if N > M else np.arange(N)

        plt.figure()
        plt.scatter(kp0_x[idx], kp0_y[idx], s=2)
        plt.xlabel("kp0_x")
        plt.ylabel("kp0_y")
        plt.title("KP0 XY coverage (subsample)")
        plt.axis("equal")
        plt.show()

        # kp0 XZ scatter
        plt.figure()
        plt.scatter(kp0_x[idx], kp0_z[idx], s=2)
        plt.xlabel("kp0_x")
        plt.ylabel("kp0_z")
        plt.title("KP0 XZ coverage (subsample)")
        plt.show()

        # yaw / orientation plots
        plt.figure()
        plt.hist(yaw_deg, bins=args.bins)
        plt.xlabel("yaw_deg of (kp1-kp0)")
        plt.ylabel("count")
        plt.title("Yaw distribution of EE +X direction (from keypoints)")
        plt.show()

        plt.figure()
        plt.hist(elev_deg, bins=args.bins)
        plt.xlabel("pitch_deg of (kp1-kp0)")
        plt.ylabel("count")
        plt.title("Pitch distribution of EE +X direction")
        plt.show()

        plt.figure()
        plt.hist(roll_deg, bins=args.bins)
        plt.xlabel("roll_deg about EE local +X")
        plt.ylabel("count")
        plt.title("Roll distribution reconstructed from EE +Z direction")
        plt.show()

        plt.figure()
        plt.hist(tilt_deg, bins=36)
        plt.xlabel("tilt_deg of (kp2-kp0) from +Z")
        plt.ylabel("count")
        plt.title("Tilt distribution of EE +Z direction")
        plt.show()

        plt.figure()
        plt.scatter(yaw_deg[idx], roll_deg[idx], s=2)
        plt.xlabel("yaw_deg (v1)")
        plt.ylabel("roll_deg (about local +X)")
        plt.title("Yaw vs roll (subsample)")
        plt.show()

        plt.figure()
        plt.scatter(elev_deg[idx], roll_deg[idx], s=2)
        plt.xlabel("pitch_deg (v1 elevation)")
        plt.ylabel("roll_deg (about local +X)")
        plt.title("Pitch vs roll (subsample)")
        plt.show()

        plt.figure()
        plt.scatter(yaw_deg[idx], tilt_deg[idx], s=2)
        plt.xlabel("yaw_deg (v1)")
        plt.ylabel("tilt_deg (v2 from +Z)")
        plt.title("Yaw vs tilt (subsample)")
        plt.show()


if __name__ == "__main__":
    main()