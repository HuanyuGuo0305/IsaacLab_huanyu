# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unsupervised Actuator Net (UAN) training environment for the Unitree Z1.

Reproduces Section II-A of "Bridging the Sim-to-Real Gap for Athletic
Loco-Manipulation" (arXiv:2502.10894) on hardware data collected by
``unitree_sdk2_python_huanyu/deploy/z1_uan_data_collection.py``.

THE IDEA
--------
Real and simulated actuators diverge, and the divergence is unlabelled: we
never measured the corrective torque, only where the real joint ended up. So
the correction is learned with RL instead of supervised regression. The
network sees a history of tracking errors and outputs a residual torque; the
reward is how closely the simulated joint then tracks the real one::

    min || f_real(s, tau) - f_sim(s, tau + pi_UAN(e)) ||

ONE STEP
--------
Each environment replays a contiguous window of real data, one sample per
physics step, and never resets mid-window::

    q_des[t]  from the log
        |
        v   nominal law, from UNITREE_Z1_UAN_CFG
    tau_nom = dcmotor_clip( Kp*(q_des[t] - q_sim) - Kd*qd_sim )
        |
        +   residual, added AFTER the clip so it can lift the joint past
        |   the nominal envelope (the paper's order)
    tau = tau_nom + action_scale * pi_UAN(error history)
        |
        v   physics
    q_sim[t+1]  vs  q_real[t+1] from the log  ->  reward

OBSERVATION
-----------
Per joint, a history of ``history_length`` (position error, velocity) pairs.
At 250 Hz, 25 steps is the paper's 100 ms. Laid out joint-major so the actor
can reshape to ``(num_envs, 6, 2 * history_length)`` and run one shared
network across all six actuators -- the paper's arrangement, which multiplies
the effective data per gradient step and stops the net keying on joint
identity.

The critic additionally sees the residual it just applied and the absolute
positions, which the actor is deliberately denied so it cannot infer pose.

SIM RATE
--------
``sim.dt`` must equal one dataset sample. The logs are 250 Hz, so dt = 1/250
and ``decimation = 1``: one policy step, one physics step, one real sample.
``UANHardwareDataset`` refuses logs recorded at any other rate rather than
letting a time-warped replay pass silently.
"""

from __future__ import annotations

import copy
import math
import os
import re

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_Z1_UAN_CFG

from .uan_dataset import NUM_ARM_JOINTS, UANHardwareDataset

# Where the collection script writes its assembled dataset. Override with
# UAN_DATASET_DIR or by editing the cfg.
_DEFAULT_DATASET_DIR = os.environ.get(
    "UAN_DATASET_DIR",
    "/home/huanyuguo/Workspace_huanyu/unitree_sdk2_python_huanyu/Log/z1_uan/dataset_20260905",
)


@configclass
class UANEnvCfg(DirectRLEnvCfg):
    """Configuration for Z1 unsupervised actuator net training."""

    # -- dataset ---------------------------------------------------------
    dataset_dir: str = _DEFAULT_DATASET_DIR
    log_files: list[str] = ["square_sine_log.pkl", "noise_log.pkl"]

    # The logs are 250 Hz; sim.dt below must be its reciprocal.
    dataset_rate_hz: float = 250.0

    # -- episode ---------------------------------------------------------
    # 20 s rollouts, as in the paper: long enough that the net has to stay
    # accurate over many steps rather than fitting a single transition.
    episode_length_s: float = 20.0
    decimation: int = 1

    # -- observation -----------------------------------------------------
    # 25 samples at 250 Hz = 100 ms, the paper's window.
    history_length: int = 25
    obs_per_joint: int = 2 * history_length          # (pos error, velocity)
    observation_space: int = NUM_ARM_JOINTS * obs_per_joint
    # Critic adds, per joint: last residual, q - q_default, q_des - q_default.
    state_space: int = observation_space + 3 * NUM_ARM_JOINTS
    action_space: int = NUM_ARM_JOINTS

    # -- action ----------------------------------------------------------
    # Residual torque = action * action_scale * nominal stiffness, so a unit
    # action is a torque worth `action_scale` rad of position error. Matches
    # the paper's `actions * action_scale * Kp`.
    action_scale: float = 0.2
    clip_actions: float | None = None

    # Hard cap on the total applied torque, as a multiple of each joint's
    # effort limit. The residual is intentionally not clipped by the nominal
    # torque-speed envelope, but it must not be unbounded either.
    applied_torque_limit_scale: float = 1.5

    # -- observation scaling ---------------------------------------------
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05

    # -- noise -----------------------------------------------------------
    add_noise: bool = True
    dof_pos_noise: float = 0.01
    dof_vel_noise: float = 1.5
    noise_level: float = 1.0

    # -- reward ----------------------------------------------------------
    # Position error between sim and real, shaped at three sharpnesses so
    # there is gradient both far away and very close in, plus an L1 term and
    # an action-rate term that biases toward gradual corrections.
    rew_survival: float = 0.0
    rew_l1: float = -1.5
    rew_exp_l2_loose: float = 4.0
    rew_exp_l2_loose_coef: float = 100.0
    rew_exp_l2: float = 4.0
    rew_exp_l2_coef: float = 300.0
    rew_exp_l2_strict: float = 5.0
    rew_exp_l2_strict_coef: float = 1000.0
    rew_action_rate: float = 0.5
    rew_action_rate_coef: float = 0.5

    # -- termination -----------------------------------------------------
    # The paper disables early termination: a rollout that diverges is
    # exactly the signal the reward needs to see.
    enable_early_termination: bool = False
    max_joint_error: float = 1.0

    # -- scene -----------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 250.0,
        render_interval=25,
        gravity=(0.0, 0.0, -9.81),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=1.5, replicate_physics=True
    )
    robot: ArticulationCfg = UNITREE_Z1_UAN_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )


class UANEnv(DirectRLEnv):
    """Learns a residual torque that makes the simulated Z1 track the real one."""

    cfg: UANEnvCfg

    def __init__(self, cfg: UANEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        n, j = self.num_envs, NUM_ARM_JOINTS
        dev = self.device

        # -- hardware data -----------------------------------------------
        window = int(round(cfg.episode_length_s * cfg.dataset_rate_hz))
        paths = [os.path.join(cfg.dataset_dir, f) for f in cfg.log_files]
        self.dataset = UANHardwareDataset(
            log_paths=paths,
            window_length=window,
            expected_rate_hz=cfg.dataset_rate_hz,
        )
        self.dataset.assert_consistent_collection()
        print("[UAN] " + self.dataset.summary().replace("\n", "\n[UAN] "))

        self.window_length = window
        self.hw_q_des = torch.from_numpy(self.dataset.q_des).to(dev)
        self.hw_q = torch.from_numpy(self.dataset.q).to(dev)
        self.hw_qd = torch.from_numpy(self.dataset.qd).to(dev)
        self.hw_starts = torch.from_numpy(self.dataset.window_starts).to(dev)

        # -- nominal actuator law, read from the articulation cfg ---------
        # One source of truth: UNITREE_Z1_UAN_CFG defines the law, this env
        # applies it. See _resolve_nominal_actuator for why PhysX does not.
        (
            self.kp,
            self.kd,
            self.effort_limit,
            self.saturation_effort,
            self.velocity_limit,
        ) = self._resolve_nominal_actuator()
        self.applied_torque_limit = (
            self.effort_limit * cfg.applied_torque_limit_scale
        )

        # -- buffers ------------------------------------------------------
        self.window_start = torch.zeros(n, dtype=torch.long, device=dev)
        self.window_t = torch.zeros(n, dtype=torch.long, device=dev)
        self.actions = torch.zeros(n, j, device=dev)
        self.prev_actions = torch.zeros(n, j, device=dev)
        self.applied_residual = torch.zeros(n, j, device=dev)
        # (num_envs, joints, history, 2) -- joint-major, newest last.
        self.err_history = torch.zeros(
            n, j, cfg.history_length, 2, device=dev
        )
        self.default_dof_pos = self._robot.data.default_joint_pos.clone()

        self._noise_scale = torch.tensor(
            [
                cfg.dof_pos_noise * cfg.noise_level * cfg.dof_pos_scale,
                cfg.dof_vel_noise * cfg.noise_level * cfg.dof_vel_scale,
            ],
            device=dev,
        )

        self._episode_sums = {
            k: torch.zeros(n, device=dev)
            for k in (
                "survival",
                "l1",
                "exp_l2_loose",
                "exp_l2",
                "exp_l2_strict",
                "action_rate",
            )
        }
        self.extras["log"] = {}

    # -- setup -----------------------------------------------------------

    def _setup_scene(self):
        self._robot = Articulation(self._passthrough_robot_cfg())
        self.scene.articulations["robot"] = self._robot
        self.scene.clone_environments(copy_from_source=False)
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    def _passthrough_robot_cfg(self) -> ArticulationCfg:
        """The same robot, but with actuators that apply effort verbatim.

        UNITREE_Z1_UAN_CFG declares the nominal law as DCMotorCfg because
        that is the spec the downstream WBC should train against. If PhysX
        were allowed to run it here, two things would go wrong at once: the
        PD law would be applied twice (once by the actuator, once by this
        env), and the residual would be routed through the actuator's
        feed-forward term and clipped by the torque-speed envelope together
        with the nominal torque.

        The residual has to be added AFTER that clip -- the paper's order,
        and the only order in which it can lift the joint past the nominal
        envelope, which it must, because the real arm was measured making
        30+ N*m at 3.9 rad/s. So the actuators are swapped for zero-gain
        implicit ones with headroom, and this env applies the law itself.
        """
        cfg = copy.deepcopy(self.cfg.robot)
        headroom = float(self.cfg.applied_torque_limit_scale)
        passthrough = {}
        for name, act in cfg.actuators.items():
            limit = act.effort_limit
            limit = max(limit) if isinstance(limit, dict) else float(limit)
            passthrough[name] = ImplicitActuatorCfg(
                joint_names_expr=list(act.joint_names_expr),
                effort_limit=limit * headroom,
                velocity_limit=None,
                stiffness=0.0,
                damping=0.0,
                friction=act.friction,
                armature=act.armature,
            )
        cfg.actuators = passthrough
        return cfg

    def _resolve_nominal_actuator(self):
        """Read the nominal PD law from the ORIGINAL articulation config.

        Deliberately from ``self.cfg.robot`` and not from the instantiated
        actuators: those were replaced with pass-through ones by
        ``_passthrough_robot_cfg``, so their gains are zero. This keeps
        UNITREE_Z1_UAN_CFG the single source of truth for the law while PhysX
        stays out of the loop.
        """
        j, dev = NUM_ARM_JOINTS, self.device
        kp = torch.full((j,), float("nan"), device=dev)
        kd = torch.zeros(j, device=dev)
        eff = torch.zeros(j, device=dev)
        sat = torch.zeros(j, device=dev)
        vel = torch.zeros(j, device=dev)

        names = list(self._robot.data.joint_names)

        def _pick(value, joint_name, default=None):
            if value is None:
                return default
            if isinstance(value, dict):
                for expr, v in value.items():
                    if re.fullmatch(expr, joint_name):
                        return float(v)
                return default
            return float(value)

        for act in self.cfg.robot.actuators.values():
            for joint_idx, joint_name in enumerate(names[:j]):
                if not any(
                    re.fullmatch(expr, joint_name) for expr in act.joint_names_expr
                ):
                    continue
                kp[joint_idx] = _pick(act.stiffness, joint_name, 0.0)
                kd[joint_idx] = _pick(act.damping, joint_name, 0.0)
                eff[joint_idx] = _pick(act.effort_limit, joint_name, 0.0)
                sat[joint_idx] = _pick(
                    getattr(act, "saturation_effort", None),
                    joint_name,
                    float(eff[joint_idx]),
                )
                vel[joint_idx] = _pick(
                    getattr(act, "velocity_limit", None), joint_name, math.inf
                )

        if torch.any(torch.isnan(kp)):
            missing = [
                names[i] for i in range(j) if bool(torch.isnan(kp[i]))
            ]
            raise ValueError(
                f"no actuator config matched joints {missing}; every arm "
                "joint needs a nominal law in UNITREE_Z1_UAN_CFG"
            )
        if torch.any(kp <= 0.0):
            raise ValueError(
                f"nominal stiffness must be positive for all six arm joints, "
                f"got {kp.tolist()} for joints {names[:j]}"
            )
        print(f"[UAN] joints            = {names[:j]}")
        print(f"[UAN] nominal Kp        = {kp.tolist()}")
        print(f"[UAN] nominal Kd        = {kd.tolist()}")
        print(f"[UAN] effort limit      = {eff.tolist()}")
        print(f"[UAN] saturation effort = {sat.tolist()}")
        print(f"[UAN] velocity limit    = {vel.tolist()}")
        return kp, kd, eff, sat, vel

    # -- indexing helpers -------------------------------------------------

    def _hw_index(self, offset: int = 0) -> torch.Tensor:
        """Absolute dataset row for each env at ``window_t + offset``."""
        t = torch.clamp(self.window_t + offset, max=self.window_length - 1)
        return self.window_start + t

    # -- stepping ---------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone()
        if self.cfg.clip_actions is not None:
            self.actions = self.actions.clamp(
                -self.cfg.clip_actions, self.cfg.clip_actions
            )

    def _apply_action(self) -> None:
        q = self._robot.data.joint_pos[:, :NUM_ARM_JOINTS]
        qd = self._robot.data.joint_vel[:, :NUM_ARM_JOINTS]
        q_des = self.hw_q_des[self._hw_index(0)]

        tau_pd = self.kp * (q_des - q) - self.kd * qd
        tau_nom = self._dcmotor_clip(tau_pd, qd)

        # After the clip, deliberately: this is the only order in which the
        # residual can push the joint past the nominal envelope.
        self.applied_residual = self.actions * self.cfg.action_scale * self.kp
        tau = tau_nom + self.applied_residual
        tau = tau.clamp(-self.applied_torque_limit, self.applied_torque_limit)

        self._robot.set_joint_effort_target(tau, joint_ids=list(range(NUM_ARM_JOINTS)))

    def _dcmotor_clip(self, effort: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
        """IsaacLab's DCMotor four-quadrant torque-speed envelope."""
        vel_at_limit = self.velocity_limit * (
            1.0 + self.effort_limit / self.saturation_effort
        )
        v = qd.clamp(-vel_at_limit, vel_at_limit)
        max_effort = torch.minimum(
            self.saturation_effort * (1.0 - v / self.velocity_limit),
            self.effort_limit,
        )
        min_effort = torch.maximum(
            self.saturation_effort * (-1.0 - v / self.velocity_limit),
            -self.effort_limit,
        )
        return torch.clamp(effort, min_effort, max_effort)

    # -- observations ------------------------------------------------------

    def _get_observations(self) -> dict:
        q = self._robot.data.joint_pos[:, :NUM_ARM_JOINTS]
        qd = self._robot.data.joint_vel[:, :NUM_ARM_JOINTS]
        q_des = self.hw_q_des[self._hw_index(0)]

        pos_err = (q_des - q) * self.cfg.dof_pos_scale
        vel = qd * self.cfg.dof_vel_scale
        sample = torch.stack((pos_err, vel), dim=-1)          # (n, j, 2)

        if self.cfg.add_noise:
            sample = sample + (
                torch.rand_like(sample) * 2.0 - 1.0
            ) * self._noise_scale

        self.err_history = torch.roll(self.err_history, shifts=-1, dims=2)
        self.err_history[:, :, -1, :] = sample

        # Joint-major flatten: the actor reshapes to (n, joints, 2*history)
        # and runs one shared network per joint.
        policy = self.err_history.reshape(self.num_envs, -1)

        critic = torch.cat(
            (
                policy,
                self.actions,
                (q - self.default_dof_pos[:, :NUM_ARM_JOINTS]) * self.cfg.dof_pos_scale,
                (q_des - self.default_dof_pos[:, :NUM_ARM_JOINTS]) * self.cfg.dof_pos_scale,
            ),
            dim=-1,
        )
        return {"policy": policy, "critic": critic}

    # -- reward -------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        q_sim = self._robot.data.joint_pos[:, :NUM_ARM_JOINTS]
        # Physics has already advanced, so compare against the NEXT real
        # sample: this is the f_real vs f_sim transition error itself.
        q_real = self.hw_q[self._hw_index(1)]

        err = q_sim - q_real
        sq = torch.sum(torch.square(err), dim=1)
        l1 = torch.sum(torch.abs(err), dim=1)
        action_rate = torch.sum(
            torch.square(self.actions - self.prev_actions), dim=1
        )

        terms = {
            "survival": torch.ones_like(sq) * cfg.rew_survival,
            "l1": l1 * cfg.rew_l1,
            "exp_l2_loose": cfg.rew_exp_l2_loose
            * torch.exp(-sq * cfg.rew_exp_l2_loose_coef),
            "exp_l2": cfg.rew_exp_l2 * torch.exp(-sq * cfg.rew_exp_l2_coef),
            "exp_l2_strict": cfg.rew_exp_l2_strict
            * torch.exp(-sq * cfg.rew_exp_l2_strict_coef),
            "action_rate": cfg.rew_action_rate
            * torch.exp(-action_rate * cfg.rew_action_rate_coef),
        }
        for k, v in terms.items():
            self._episode_sums[k] += v

        # Advance only after the reward has consumed sample t+1, so
        # observations below are built from q_des at the new step.
        self.window_t += 1
        return torch.sum(torch.stack(list(terms.values())), dim=0) * self.step_dt

    # -- termination ---------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # One sample of headroom: the reward reads window_t + 1.
        time_out = self.window_t >= (self.window_length - 2)

        if self.cfg.enable_early_termination:
            q_sim = self._robot.data.joint_pos[:, :NUM_ARM_JOINTS]
            q_real = self.hw_q[self._hw_index(1)]
            died = torch.any(
                torch.abs(q_sim - q_real) > self.cfg.max_joint_error, dim=1
            )
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    # -- reset ----------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # A fresh window per env, uniform over every valid start. Starts never
        # straddle two logs -- see UANHardwareDataset.
        pick = torch.randint(
            0, self.hw_starts.shape[0], (len(env_ids),), device=self.device
        )
        self.window_start[env_ids] = self.hw_starts[pick]
        self.window_t[env_ids] = 0

        # Put the simulated arm exactly where the real one was at the window
        # start, so the rollout begins from a real state rather than a guess.
        row = self.window_start[env_ids]
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        joint_pos[:, :NUM_ARM_JOINTS] = self.hw_q[row]
        joint_vel[:, :NUM_ARM_JOINTS] = self.hw_qd[row]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.applied_residual[env_ids] = 0.0
        self.err_history[env_ids] = 0.0

        extras = {}
        for key, buf in self._episode_sums.items():
            extras[f"Episode_Reward/{key}"] = (
                torch.mean(buf[env_ids]) / self.max_episode_length_s
            )
            buf[env_ids] = 0.0
        self.extras["log"] = dict(extras)
