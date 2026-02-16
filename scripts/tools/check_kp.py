"""
Offline check for keypoints-table orientation distribution.

Input: .npy with shape (N,9): [kp0(3), kp1(3), kp2(3)]
Assumes:
  v1 = kp1 - kp0  (EE +X direction * dx)
  v2 = kp2 - kp0  (EE +Z direction * dz)

Outputs:
- Sanity checks: ||v1||, ||v2||, orthogonality
- Orientation distributions:
    * yaw of v1 in XY plane: atan2(v1_y, v1_x)  (deg)
    * pitch of v1 (elevation): atan2(v1_z, sqrt(v1_x^2+v1_y^2)) (deg)
    * tilt of v2 from +Z: acos(v2_z/||v2||) (deg)
    * v2_z histogram (how "up" the EE +Z is)
- Coverage stats: percent with v1_x>0, v2_z>0, etc.
- Optional scatter plots (matplotlib) if --plot is set.

Usage:
  python check_kp.py --file scripts/tools/reachable_kp0kp1kp2_lb.npy --plot
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, required=True, help="Path to kp table .npy (N,9).")
    ap.add_argument("--dx", type=float, default=0.30, help="Expected ||kp1-kp0||.")
    ap.add_argument("--dz", type=float, default=0.30, help="Expected ||kp2-kp0||.")
    ap.add_argument("--tol", type=float, default=5e-3, help="Tolerance for norm sanity checks.")
    ap.add_argument("--bins", type=int, default=36, help="Histogram bins for yaw/angles.")
    ap.add_argument("--plot", action="store_true", help="Show matplotlib plots.")
    args = ap.parse_args()

    arr = np.load(args.file).astype(np.float64)
    if arr.ndim != 2 or arr.shape[1] != 9:
        raise ValueError(f"Expected (N,9), got {arr.shape}")

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

    # v1 yaw in XY (rad->deg)
    yaw = np.arctan2(v1u[:, 1], v1u[:, 0])
    yaw = _wrap_pi(yaw)
    yaw_deg = np.degrees(yaw)

    # v1 elevation (pitch) relative to XY plane
    xy = np.sqrt(v1u[:, 0] ** 2 + v1u[:, 1] ** 2)
    elev = np.arctan2(v1u[:, 2], xy)
    elev_deg = np.degrees(elev)

    # v2 tilt from +Z
    v2z = np.clip(v2u[:, 2], -1.0, 1.0)
    tilt = np.arccos(v2z)             # 0 means perfectly up
    tilt_deg = np.degrees(tilt)

    # Summaries
    N = arr.shape[0]
    print(f"\n[FILE] {args.file}")
    print(f"[N] {N}")

    print("\n--- Sanity checks ---")
    print(f"||v1|| mean={n1.mean():.6f}, std={n1.std():.6f}, min={n1.min():.6f}, max={n1.max():.6f}  (expect ~{args.dx})")
    print(f"||v2|| mean={n2.mean():.6f}, std={n2.std():.6f}, min={n2.min():.6f}, max={n2.max():.6f}  (expect ~{args.dz})")
    print(f"cos(v1,v2) mean={cos_12.mean():.3e}, std={cos_12.std():.3e}, max_abs={np.max(np.abs(cos_12)):.3e}")

    ok_n1 = np.abs(n1 - args.dx) < args.tol
    ok_n2 = np.abs(n2 - args.dz) < args.tol
    ok_ortho = np.abs(cos_12) < 5e-3
    print(f"pass ||v1|| within tol: {_percent(ok_n1):.2f}%")
    print(f"pass ||v2|| within tol: {_percent(ok_n2):.2f}%")
    print(f"pass orthogonality |cos|<5e-3: {_percent(ok_ortho):.2f}%")

    print("\n--- Coverage / constraints style stats ---")
    print(f"v1_x > 0 (EE +X in front half-space): {_percent(v1u[:,0] > 0):.2f}%")
    print(f"v2_z > 0 (EE +Z points upward-ish): {_percent(v2u[:,2] > 0):.2f}%")
    print(f"tilt < 30 deg (Z within 30deg of up): {_percent(tilt_deg < 30.0):.2f}%")
    print(f"tilt < 60 deg: {_percent(tilt_deg < 60.0):.2f}%")

    print("\n--- Orientation distributions ---")
    def pr(name, x):
        qs = np.percentile(x, [0, 1, 5, 25, 50, 75, 95, 99, 100])
        print(f"{name}: mean={x.mean():.2f}, std={x.std():.2f}, "
              f"p1={qs[1]:.2f}, p5={qs[2]:.2f}, p50={qs[4]:.2f}, p95={qs[6]:.2f}, p99={qs[7]:.2f}")

    pr("yaw_deg(v1 in XY)", yaw_deg)
    pr("elev_deg(v1)", elev_deg)
    pr("tilt_deg(v2 from +Z)", tilt_deg)

    # Yaw histogram occupancy
    hist, edges = _hist_summary_deg(yaw_deg, bins=args.bins)
    occ = np.sum(hist > 0)
    print(f"\nYaw histogram bins: {args.bins}, occupied bins: {occ}/{args.bins} ({100.0*occ/args.bins:.1f}%)")
    top = np.argsort(hist)[::-1][:10]
    print("Top yaw bins (count, range_deg):")
    for i in top:
        if hist[i] == 0:
            break
        lo, hi = edges[i], edges[i+1]
        print(f"  {hist[i]:6d}  [{lo:7.1f}, {hi:7.1f})")

    # Tilt histogram (0..180)
    tilt_hist, tilt_edges = np.histogram(tilt_deg, bins=36, range=(0.0, 180.0))
    occ2 = np.sum(tilt_hist > 0)
    print(f"\nTilt histogram bins: 36 (0..180), occupied: {occ2}/36 ({100.0*occ2/36:.1f}%)")
    top2 = np.argsort(tilt_hist)[::-1][:10]
    print("Top tilt bins (count, range_deg):")
    for i in top2:
        if tilt_hist[i] == 0:
            break
        lo, hi = tilt_edges[i], tilt_edges[i+1]
        print(f"  {tilt_hist[i]:6d}  [{lo:7.1f}, {hi:7.1f})")

    # Optional plots
    if args.plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.hist(yaw_deg, bins=args.bins)
        plt.xlabel("yaw_deg of (kp1-kp0)")
        plt.ylabel("count")
        plt.title("Yaw distribution of EE +X direction (from keypoints)")
        plt.show()

        plt.figure()
        plt.hist(elev_deg, bins=36)
        plt.xlabel("elevation_deg of (kp1-kp0)")
        plt.ylabel("count")
        plt.title("Elevation distribution of EE +X direction")
        plt.show()

        plt.figure()
        plt.hist(tilt_deg, bins=36)
        plt.xlabel("tilt_deg of (kp2-kp0) from +Z")
        plt.ylabel("count")
        plt.title("Tilt distribution of EE +Z direction")
        plt.show()

        # Scatter: yaw vs tilt (subsample for speed)
        M = min(N, 20000)
        idx = np.random.default_rng(0).choice(N, size=M, replace=False) if N > M else np.arange(N)
        plt.figure()
        plt.scatter(yaw_deg[idx], tilt_deg[idx], s=2)
        plt.xlabel("yaw_deg (v1)")
        plt.ylabel("tilt_deg (v2 from +Z)")
        plt.title("Yaw vs tilt (subsample)")
        plt.show()

if __name__ == "__main__":
    main()
