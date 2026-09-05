# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--reward_visualizer",
    action="store_true",
    default=False,
    help="Display real-time curves for the environment reward terms during policy inference.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
from tensordict import TensorDictBase

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

def _to_cpu_numpy(x):
    """Convert tensor-like data to CPU numpy for printing."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def print_first_env_actor_io(obs, actions, step, start_step=5, end_step=10):
    """Print only the first environment's actor input and output."""
    if step < start_step or step > end_step:
        return

    print(f"\n================ ACTOR DEBUG STEP {step} ================")

    # only cares about actor obs
    if isinstance(obs, dict):
        print("[ACTOR OBS] dict keys:", list(obs.keys()))
        for k, v in obs.items():
            key_str = str(k).lower()
            if key_str in ["critic", "states", "privileged_obs", "critic_obs"]:
                continue
            v_np = _to_cpu_numpy(v)
            if hasattr(v_np, "shape"):
                print(f"[ACTOR OBS] {k}: shape={v_np.shape}")
                if len(v_np.shape) >= 1:
                    print(f"[ACTOR OBS] {k}[0] = {v_np[0]}")
            else:
                print(f"[ACTOR OBS] {k} = {v_np}")

    elif isinstance(obs, TensorDictBase):
        print("[ACTOR OBS] tensordict keys:", list(obs.keys()))
        for k in obs.keys():
            key_str = str(k).lower()
            if key_str in ["critic", "states", "privileged_obs", "critic_obs"]:
                continue
            v = obs[k]
            v_np = _to_cpu_numpy(v)
            if hasattr(v_np, "shape"):
                print(f"[ACTOR OBS] {k}: shape={v_np.shape}")
                if len(v_np.shape) >= 1:
                    print(f"[ACTOR OBS] {k}[0] = {v_np[0]}")
            else:
                print(f"[ACTOR OBS] {k} = {v_np}")

    else:
        obs_np = _to_cpu_numpy(obs)
        if hasattr(obs_np, "shape"):
            print(f"[ACTOR OBS] shape = {obs_np.shape}")
            print(f"[ACTOR OBS] first env = {obs_np[0]}")
        else:
            print(f"[ACTOR OBS] = {obs_np}")

    # only print actor output action
    act_np = _to_cpu_numpy(actions)
    if hasattr(act_np, "shape"):
        print(f"[ACTOR ACT] shape = {act_np.shape}")
        print(f"[ACTOR ACT] first env = {act_np[0]}")
    else:
        print(f"[ACTOR ACT] = {act_np}")

    print("=========================================================\n")


def _scalarize_reward_value(value):
    """Convert a reward log value to one float.

    Tensor/array values with an environment dimension are averaged so the
    visualizer remains readable when play.py runs with multiple environments.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean().cpu().item())

    # NumPy is intentionally not imported globally. Matplotlib depends on it,
    # and array-like values usually expose ``mean`` and ``item``.
    if hasattr(value, "mean") and hasattr(value, "item"):
        try:
            return float(value.mean().item())
        except (TypeError, ValueError, AttributeError):
            pass

    if isinstance(value, (int, float, bool)):
        return float(value)

    return None


def _collect_reward_scalars(source, prefix=""):
    """Recursively collect scalar entries whose path represents a reward term."""
    result = {}
    if source is None:
        return result

    if isinstance(source, TensorDictBase):
        source = {key: source[key] for key in source.keys()}

    if isinstance(source, dict):
        for key, value in source.items():
            key = str(key)
            path = f"{prefix}/{key}" if prefix else key
            if isinstance(value, (dict, TensorDictBase)):
                result.update(_collect_reward_scalars(value, path))
                continue

            # Environment logging convention is usually Reward/<term>.
            # Also accept reward_* and *_reward for custom environments.
            path_lower = path.lower()
            if (
                path_lower.startswith("reward/")
                or "/reward/" in path_lower
                or path_lower.startswith("rewards/")
                or "/rewards/" in path_lower
                or key.lower().startswith("reward_")
                or key.lower().endswith("_reward")
            ):
                scalar = _scalarize_reward_value(value)
                if scalar is not None:
                    result[path] = scalar

    return result


class RewardVisualizer:
    """Non-blocking Matplotlib visualizer with one subplot per reward term."""

    def __init__(self, history_length=500, refresh_interval=2, max_columns=3):
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "--reward_visualizer requires matplotlib. "
                "Install it in the Isaac Lab Python environment."
            ) from exc

        self.plt = plt
        self.history_length = int(history_length)
        self.refresh_interval = max(1, int(refresh_interval))
        self.max_columns = max(1, int(max_columns))
        self.step = 0
        self.series = {}
        self.axes = {}
        self.lines = {}
        self._layout_names = []

        self.plt.ion()
        self.fig = self.plt.figure(num="Isaac Lab Reward Visualizer", figsize=(14, 8))
        self.fig.suptitle("Real-time reward terms (mean across environments)")
        self.fig.show()

    @staticmethod
    def _short_title(name):
        """Return a compact subplot title while keeping reward names identifiable."""
        for prefix in ("Reward/", "Rewards/", "reward/", "rewards/"):
            if name.startswith(prefix):
                return name[len(prefix):]
        return name

    def _rebuild_layout(self):
        """Rebuild the subplot grid when new reward terms appear."""
        names = sorted(self.series)
        if names == self._layout_names:
            return

        self._layout_names = names
        self.fig.clear()
        self.fig.suptitle("Real-time reward terms (mean across environments)")
        self.axes.clear()
        self.lines.clear()

        count = len(names)
        if count == 0:
            self.fig.canvas.draw_idle()
            return

        columns = min(self.max_columns, count)
        rows = (count + columns - 1) // columns
        self.fig.set_size_inches(max(10.0, 4.5 * columns), max(6.0, 3.0 * rows), forward=True)

        axes = self.fig.subplots(rows, columns, squeeze=False)
        flat_axes = axes.ravel()

        for index, name in enumerate(names):
            ax = flat_axes[index]
            ax.set_title(self._short_title(name), fontsize=9)
            ax.set_xlabel("Step", fontsize=8)
            ax.set_ylabel("Reward", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", labelsize=8)
            self.axes[name] = ax
            (self.lines[name],) = ax.plot([], [], linewidth=1.2)

        for index in range(count, len(flat_axes)):
            flat_axes[index].set_visible(False)

        self.fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    def update(self, reward_values):
        """Append one sample and refresh all reward subplots without blocking Isaac Sim."""
        if not reward_values:
            return

        self.step += 1
        all_names = set(self.series) | set(reward_values)

        for name in sorted(all_names):
            if name not in self.series:
                # Backfill newly discovered terms so all plots share the same step index.
                self.series[name] = [float("nan")] * (self.step - 1)
            self.series[name].append(reward_values.get(name, float("nan")))
            if len(self.series[name]) > self.history_length:
                self.series[name] = self.series[name][-self.history_length :]

        self._rebuild_layout()

        if self.step % self.refresh_interval != 0:
            return

        for name, values in sorted(self.series.items()):
            start_step = max(0, self.step - len(values))
            x_values = list(range(start_step, self.step))
            self.lines[name].set_data(x_values, values)
            self.axes[name].relim()
            self.axes[name].autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.plt.pause(0.001)

    def close(self):
        """Close the Matplotlib window."""
        if hasattr(self, "fig"):
            self.plt.close(self.fig)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    reward_visualizer = None
    if args_cli.reward_visualizer:
        reward_visualizer = RewardVisualizer(history_length=500, refresh_interval=2)
        print("[INFO] Reward visualizer enabled.")

    # reset environment
    obs = env.get_observations()
    timestep = 0
    debug_step = 0

    # Get the joints' order for sim2sim deployment
    isaac_env = env.unwrapped
    robot = None
    if hasattr(isaac_env.scene, "articulations"):
        robot = list(isaac_env.scene.articulations.values())[0]

    if robot is not None:
        print("\n================ ISAACLAB JOINT INFORMATION ================")
        print("[INFO] Number of DOFs:", robot.num_joints)
        print("[INFO] DOF names (order used by RL):")
        for i, name in enumerate(robot.joint_names):
            print(f"  {i:2d}: {name}")
        print("============================================================\n")
    else:
        print("[WARNING] Could not find robot object to print DOF names.")

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)

            # debug: print actor obs/action for env0 on steps 5~10
            try:
                print_first_env_actor_io(obs, actions, debug_step, start_step=5, end_step=10)
            except Exception as e:
                print(f"[DEBUG] print_first_env_actor_io failed at step {debug_step}: {e}")

            # env stepping
            obs, rewards, dones, infos = env.step(actions)

            if reward_visualizer is not None:
                # Prefer step-returned info. Some DirectRLEnv implementations
                # expose the same dictionary through ``unwrapped.extras``.
                reward_values = _collect_reward_scalars(infos)
                if not reward_values:
                    reward_values = _collect_reward_scalars(
                        getattr(env.unwrapped, "extras", None)
                    )

                # Always include the total per-step reward returned by the wrapper.
                total_reward = _scalarize_reward_value(rewards)
                if total_reward is not None:
                    reward_values["Reward/total_step_reward"] = total_reward

                reward_visualizer.update(reward_values)

        debug_step += 1

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    if reward_visualizer is not None:
        reward_visualizer.close()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()