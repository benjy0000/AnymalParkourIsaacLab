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


def _collect_step_terminations(tm):
    """
    Use tm.term_dones: dict[name] -> BoolTensor[num_envs] indicating which envs
    terminated for each term THIS step.
    Returns dict[name] -> list[int] env_ids.
    """
    causes: dict[str, list[int]] = {}
    term_dones = getattr(tm, "_term_dones", None)
    if not term_dones:
        return causes
    for name, done_mask in term_dones.items():
        if done_mask is None:
            continue
        # Expect torch.BoolTensor [num_envs]
        if hasattr(done_mask, "nonzero"):
            env_ids = done_mask.nonzero(as_tuple=False).flatten().tolist()
            if env_ids:
                causes[name] = [int(i) for i in env_ids]
    return causes


def main():
    """Evaluate with RSL-RL agent."""
    # determine if barkour score is being calculated
    is_barkour = "barkour" in args_cli.task
    
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

    # get terminations manager
    tm = getattr(env.unwrapped, "termination_manager", None)

    # termination cause -> count
    acc_term_counts: dict[str, int] = {}
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

            # print infos if available
            any_cause = False
            if tm is not None:
                step_causes = _collect_step_terminations(tm)
                for cause, env_ids in step_causes.items():
                    c = len(env_ids)
                    if c > 0:
                        if cause in acc_term_counts:
                            acc_term_counts[cause] += c
                        else:
                            acc_term_counts[cause] = c
                        total_terminations += c
                        any_cause = True

            if any_cause and total_terminations > 0:
                print("\n[EVAL] Termination summary (direct from termination manager):")
                for cause in sorted(acc_term_counts):
                    count = acc_term_counts[cause]
                    pct = 100.0 * count / total_terminations
                    print(f"  - {cause}: {pct:.1f}% (n={count})")
                print()

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