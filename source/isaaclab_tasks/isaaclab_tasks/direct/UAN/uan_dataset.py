# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hardware dataset for Unsupervised Actuator Net (UAN) training.

Loads the logs written by
``unitree_sdk2_python_huanyu/deploy/z1_uan_data_collection.py`` and serves
fixed-length windows of real transitions to the training environment.

The log schema is the one from "Bridging the Sim-to-Real Gap for Athletic
Loco-Manipulation" (arXiv:2502.10894), so these files also load unmodified
through that repo's ``load_hardware_data``::

    data['arm_pd_tau_targets']['q_des']       (N, 6)   commanded position
    data['arm_pd_tau_targets']['gripperQ_des'](N,)
    data['arm_pd_tau_targets']['kp'] / ['kd'] (N, 7)   firmware gains
    data['arm_pd_tau_targets']['timestamp']   (N,)     microseconds
    data['arm_control_data']['q'] / ['qd'] / ['tau_est']  (N, 7)  [j1..j6, gripper]
    data['arm_control_data']['timestamp']     (N,)     microseconds
    data['uan_meta']                          collection metadata

Only the six arm joints are used; column 6 (the gripper) is dropped, since
the gripper was held at a fixed target throughout collection and is not
modelled by the UAN.

WHAT THE GAINS IN THE LOG ARE NOT
---------------------------------
``kp``/``kd`` in the log are the Z1 **firmware** numbers and are NOT in
N*m/rad. Measured stiffness on the real arm came to
[40.8, 100.4, 51.8, 46.9, 38.7, 31.4] N*m/rad, i.e. 12.6-22.3x the firmware
value. They are loaded for provenance and deliberately never used to
reconstruct torque: the simulator applies its own nominal PD and the UAN
learns the residual, exactly as the paper does (its released configs replay
hardware ``q_des`` through sim gains of 64/128 and discard the logged ones).

WINDOWS
-------
A window never crosses a file boundary, because two logs are unrelated
trajectories and a rollout that straddles them would be learning a
discontinuity that never happens on hardware. Windows may cross *segment*
boundaries inside a file; those are contiguous real motion.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field

import numpy as np

NUM_ARM_JOINTS = 6


@dataclass
class TrajectoryFile:
    """One hardware log."""

    path: str
    name: str
    num_samples: int
    rate_hz: float
    meta: dict = field(default_factory=dict)


class UANHardwareDataset:
    """Real Z1 transitions, served as fixed-length windows.

    Args:
        log_paths: ``.pkl`` logs to load, in order.
        window_length: samples per rollout window. At 250 Hz, 5000 = 20 s.
        expected_rate_hz: rate the logs must have been recorded at. The
            simulator steps once per sample, so ``sim_dt`` must be
            ``1 / expected_rate_hz`` or the replay is silently time-warped.
        rate_tolerance: allowed fractional deviation of the measured rate.
    """

    def __init__(
        self,
        log_paths: list[str],
        window_length: int,
        expected_rate_hz: float = 250.0,
        rate_tolerance: float = 0.02,
    ) -> None:
        if not log_paths:
            raise ValueError("UANHardwareDataset needs at least one log path.")
        if window_length < 2:
            raise ValueError("window_length must be at least 2 samples.")

        self.window_length = int(window_length)
        self.expected_rate_hz = float(expected_rate_hz)

        q_des_list: list[np.ndarray] = []
        q_list: list[np.ndarray] = []
        qd_list: list[np.ndarray] = []
        tau_list: list[np.ndarray] = []
        self.files: list[TrajectoryFile] = []
        self._starts: list[np.ndarray] = []

        offset = 0
        for path in log_paths:
            path = os.path.abspath(os.path.expanduser(path))
            if not os.path.isfile(path):
                raise FileNotFoundError(f"UAN log not found: {path}")
            with open(path, "rb") as f:
                data = pickle.load(f)

            q_des = np.asarray(
                data["arm_pd_tau_targets"]["q_des"], dtype=np.float32
            )[:, :NUM_ARM_JOINTS]
            q = np.asarray(data["arm_control_data"]["q"], dtype=np.float32)[
                :, :NUM_ARM_JOINTS
            ]
            qd = np.asarray(data["arm_control_data"]["qd"], dtype=np.float32)[
                :, :NUM_ARM_JOINTS
            ]
            tau = np.asarray(
                data["arm_control_data"]["tau_est"], dtype=np.float32
            )[:, :NUM_ARM_JOINTS]
            t_us = np.asarray(
                data["arm_control_data"]["timestamp"], dtype=np.float64
            )

            n = int(q.shape[0])
            if not (q_des.shape[0] == qd.shape[0] == tau.shape[0] == n):
                raise ValueError(f"{path}: inconsistent array lengths.")
            if n < self.window_length:
                raise ValueError(
                    f"{path}: {n} samples is shorter than the "
                    f"{self.window_length}-sample window."
                )

            # A time-warped replay is a silent, fatal error: the simulator
            # steps once per sample, so the log rate IS the sim rate.
            duration_s = float(t_us[-1] - t_us[0]) * 1e-6
            rate = (n - 1) / max(duration_s, 1e-9)
            if abs(rate - self.expected_rate_hz) > rate_tolerance * self.expected_rate_hz:
                raise ValueError(
                    f"{path}: recorded at {rate:.1f} Hz but the environment "
                    f"expects {self.expected_rate_hz:.1f} Hz. Set sim.dt to "
                    f"1/{rate:.0f} or recollect."
                )

            # Window starts that stay inside THIS file. The last usable start
            # leaves window_length samples, and the reward compares against
            # sample t+1, so the final start is n - window_length.
            self._starts.append(
                offset + np.arange(0, n - self.window_length + 1, dtype=np.int64)
            )

            q_des_list.append(q_des)
            q_list.append(q)
            qd_list.append(qd)
            tau_list.append(tau)
            self.files.append(
                TrajectoryFile(
                    path=path,
                    name=os.path.splitext(os.path.basename(path))[0],
                    num_samples=n,
                    rate_hz=rate,
                    meta=data.get("uan_meta", {}),
                )
            )
            offset += n

        self.q_des = np.concatenate(q_des_list, axis=0)
        self.q = np.concatenate(q_list, axis=0)
        self.qd = np.concatenate(qd_list, axis=0)
        self.tau_est = np.concatenate(tau_list, axis=0)
        self.window_starts = np.concatenate(self._starts, axis=0)
        self.num_samples = int(self.q.shape[0])

    # -- sampling -------------------------------------------------------

    def sample_window_starts(
        self, num: int, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Uniformly sample ``num`` window starts that stay inside one file."""
        rng = rng or np.random.default_rng()
        idx = rng.integers(0, self.window_starts.shape[0], size=int(num))
        return self.window_starts[idx]

    # -- reporting ------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"UAN dataset: {self.num_samples:,} samples "
            f"({self.num_samples / self.expected_rate_hz / 60.0:.1f} min) "
            f"from {len(self.files)} log(s)",
            f"  window {self.window_length} samples "
            f"({self.window_length / self.expected_rate_hz:.1f} s), "
            f"{self.window_starts.shape[0]:,} valid starts",
        ]
        for f in self.files:
            gains = f.meta.get("arm_kps_runtime", "?")
            lines.append(
                f"  {f.name:<18s} {f.num_samples:>8,} samples "
                f"@ {f.rate_hz:6.1f} Hz  firmware kp={gains}"
            )
        lines.append(
            f"  q range   {np.round(self.q.max(0) - self.q.min(0), 2)} rad"
        )
        lines.append(f"  |qd| max  {np.round(np.abs(self.qd).max(0), 2)} rad/s")
        lines.append(
            f"  |tau| max {np.round(np.abs(self.tau_est).max(0), 1)} Nm"
        )
        return "\n".join(lines)

    def assert_consistent_collection(self) -> None:
        """Warn if the logs were not collected under the same conditions.

        Mixing logs recorded under different firmware gains means the dataset
        describes two different plants, and the UAN would average them.
        """
        gains = [
            (
                tuple(f.meta.get("arm_kps_runtime", ())),
                tuple(f.meta.get("arm_kds_runtime", ())),
            )
            for f in self.files
        ]
        if len(set(gains)) > 1:
            detail = "\n".join(
                f"    {f.name}: kp={g[0]} kd={g[1]}"
                for f, g in zip(self.files, gains)
            )
            raise ValueError(
                "the logs were collected under DIFFERENT firmware gains, so "
                "they describe different plants and the UAN would average "
                "them:\n" + detail
            )
