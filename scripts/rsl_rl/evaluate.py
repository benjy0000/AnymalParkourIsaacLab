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
import numpy as np

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to set up environment tasks
import anymal_parkour.tasks  # noqa: F401


# Buffers for each env_id
difficulties_buffer : torch.Tensor
ep_lengths_buffer : torch.Tensor
target_speeds_buffer : torch.Tensor


def _collect_step_termination_records(tm, env):
    """Return a list of per-env termination records for THIS step."""
    
    global difficulties_buffer
    global ep_lengths_buffer
    global target_speeds_buffer

    records: list[dict] = []
    term_dones = getattr(tm, "_term_dones", None)
    if not term_dones:
        return records

    # Resolve difficulty fetch function (if available)
    terrain = getattr(getattr(env.unwrapped, "scene", None), "terrain", None)
    fetch_fn = getattr(terrain, "fetch_difficulties_from_env", None) if terrain else None

    for term_name, done_mask in term_dones.items():
        if done_mask is None or not hasattr(done_mask, "nonzero"):
            continue
        env_ids_tensor = done_mask.nonzero(as_tuple=False).flatten()
        if env_ids_tensor.numel() == 0:
            continue


        # Round to nearest 0.05 as terrain generator can slightly vary difficulties
        diffs = torch.round(difficulties_buffer[env_ids_tensor] / 0.05) * 0.05
        # Ensure 1-D list
        diffs_list = diffs.view(-1).tolist()

        # Fetch episode lengths and convert to time
        ep_lengths = ep_lengths_buffer[env_ids_tensor] * env.unwrapped.step_dt
        ep_lengths_list = ep_lengths.view(-1).tolist()

        # Fetch target speeds
        target_speeds = target_speeds_buffer[env_ids_tensor]
        target_speeds_list = target_speeds.view(-1).tolist()
        for term_num in range(len(env_ids_tensor)):
            records.append(
                {
                    "termination_type": term_name,
                    "difficulty": diffs_list[term_num],
                    "elapsed_time": ep_lengths_list[term_num],
                    "target_speed": target_speeds_list[term_num]
                }
            )

        # Update the buffers
        difficulties_buffer[env_ids_tensor] = fetch_fn(env_ids_tensor)
        ep_lengths_buffer[env_ids_tensor] = env.episode_length_buf[env_ids_tensor]
        target_speeds_buffer[env_ids_tensor] = env.unwrapped.command_manager.get_term("target_speed").command[env_ids_tensor]

    return records


def main():

    global difficulties_buffer
    global ep_lengths_buffer
    global target_speeds_buffer

    """Evaluate with RSL-RL agent."""
    # determine if barkour score is being calculated
    is_barkour = "Barkour" in args_cli.task
    
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

    # initialize necessary buffers
    terrain = getattr(getattr(env.unwrapped, "scene", None), "terrain", None)
    fetch_fn = getattr(terrain, "fetch_difficulties_from_env", None) if terrain else None
    difficulties_buffer = fetch_fn(torch.arange(env_cfg.scene.num_envs))
    target_speeds_buffer = env.unwrapped.command_manager.get_term("target_speed").command.clone()
    ep_lengths_buffer = env.episode_length_buf.clone()

    barkour_scores = []

    # termination counts broken down by difficulty -> cause -> count
    acc_term_counts_by_diff: dict[float | None, dict[str, int]] = {}
    total_terms_by_diff: dict[float | None, int] = {}

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # update episode length buffer to prevent loosing info on reset
            ep_lengths_buffer = env.episode_length_buf.clone()
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, infos = env.step(actions)

            any_terminations = False
            if tm is not None:
                # Detailed per-env termination records for THIS step
                term_records = _collect_step_termination_records(tm, env.unwrapped)
                if term_records:
                    any_terminations = True
                # Collect termination records classified by difficulty and cause
                for rec in term_records:
                    cause = rec["termination_type"]
                    diff = rec["difficulty"]  # may be float or None
                    if diff not in acc_term_counts_by_diff:
                        acc_term_counts_by_diff[diff] = {}
                        total_terms_by_diff[diff] = 0
                    acc_term_counts_by_diff[diff][cause] = acc_term_counts_by_diff[diff].get(cause, 0) + 1
                    total_terms_by_diff[diff] += 1

            if any_terminations:
                if is_barkour:
                    for rec in term_records:
                        if rec["termination_type"] == "reached_finish":
                            expected_time = 18 / rec["target_speed"]
                            actual_time = rec["elapsed_time"]
                            run_score = np.clip(1.0 - abs(expected_time - actual_time) * 0.01, a_min=0.0, a_max=1.0)
                        else:
                            run_score = 0.0
                        barkour_scores.append(run_score)
                        print("\n[EVAL] Barkour score:")
                        print(f"  Last run score: {run_score:.3f}")
                        print(f"  Average score: {np.mean(barkour_scores):.3f} (n={len(barkour_scores)})")

                else:
                    print("\n[EVAL] Termination summary by difficulty (cumulative):")
                    # Sort difficulties: None last
                    for diff in sorted([d for d in total_terms_by_diff.keys() if d is not None]) + ([None] if None in total_terms_by_diff else []):
                        total_d = total_terms_by_diff[diff]
                        label = "None" if diff is None else f"{diff:.2f}"
                        print(f"  Difficulty {label} (total n={total_d}):")
                        causes_dict = acc_term_counts_by_diff[diff]
                        # Sort by descending count
                        for cause, count in sorted(causes_dict.items(), key=lambda x: x[1], reverse=True):
                            pct = 100.0 * count / total_d if total_d > 0 else 0.0
                            print(f"    - {cause}: {pct:.1f}% (n={count})")
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