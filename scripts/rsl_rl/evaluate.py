"""Script to evaluate a checkpoint if an RL agent from RSL-RL. It should use eval_env_cfg task
   which can be configured to customise the terrain."""

"""Launch Isaac Sim Simulator first."""

import argparse

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
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import statistics

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to set up environment tasks
import anymal_parkour.tasks  # noqa: F401


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
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

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # in-sim step dt
    step_dt = float(getattr(env.unwrapped, "step_dt", 0.0))
    episode_time_s = 0.0
    # termination cause -> list of episode lengths (seconds)
    term_durations: dict[str, list[float]] = {}
    # termination cause -> count
    term_counts: dict[str, int] = {}
    total_terminations = 0

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, infos = env.step(actions)
        # advance in-sim time
        episode_time_s += step_dt

        # detect termination (note there should only be one env)
        done = bool(dones.item()) if isinstance(dones, torch.Tensor) else bool(dones)
        if done:
            any_cause = False
            # scrape termination flags named 'Episode_Termination/...'
            if isinstance(infos["log"], dict):
                for key, val in infos["log"].items():
                    if isinstance(key, str) and key.startswith("Episode_Termination/"):
                        key = key.removeprefix("Episode_Termination/")
                        # get scalar flag (0/1)
                        if isinstance(val, (int, float, bool)):
                            flag = int(val)
                        else:
                            continue
                        if flag == 1:
                            any_cause = True
                            term_durations.setdefault(key, []).append(episode_time_s)
                            term_counts[key] = term_counts.get(key, 0) + 1
                            break  # only one cause per episode

            if any_cause:
                total_terminations += 1

                # print running stats: percentage per cause and mean episode length
                print("\n[EVAL] Termination summary:")
                for cause in sorted(term_counts.keys()):
                    count = term_counts[cause]
                    pct = (100.0 * count / total_terminations) if total_terminations > 0 else 0.0
                    mean_len = statistics.mean(term_durations[cause]) if term_durations[cause] else 0.0
                    print(f"  - {cause}: {pct:.1f}%  |  mean_ep_len={mean_len:.2f}s  (n={count})")
                print()

            # reset episode timer after termination
            episode_time_s = 0.0

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()