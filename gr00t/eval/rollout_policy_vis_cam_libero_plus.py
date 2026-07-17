# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path
import sys
import time
from typing import Any
import uuid

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.sim.env_utils import get_embodiment_tag_from_env_name
from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
from gr00t.policy import BasePolicy
from gr00t.utils.determinism import seed_everything
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import tyro


ROBOCASA_PANDA_RECORD_VIDEO_KEYS = (
    "video.res256_image_side_0",
    "video.res256_image_side_1",
    "video.res256_image_wrist_0",
)


class TrtMode(str, Enum):
    """TensorRT inference modes."""

    N17_FULL_PIPELINE = "n17_full_pipeline"
    VIT_LLM_ONLY = "vit_llm_only"
    ACTION_HEAD = "action_head"


@dataclass
class VideoConfig:
    """Configuration for video recording settings.

    Attributes:
        video_dir: Directory to save videos (if None, no videos are saved)
        steps_per_render: Number of steps between each call to env.render() while recording
            during rollout
        fps: Frames per second for the output video
        codec: Video codec to use for compression
        input_pix_fmt: Input pixel format
        crf: Constant Rate Factor for video compression (lower = better quality)
        thread_type: Threading strategy for video encoding
        thread_count: Number of threads to use for encoding
    """

    video_dir: str | None = None
    steps_per_render: int = 2
    max_episode_steps: int = 720
    fps: int = 20
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 1
    overlay_text: bool = True
    n_action_steps: int = 8
    record_video_keys: tuple[str, ...] | None = None


@dataclass
class MultiStepConfig:
    """Configuration for multi-step environment settings.

    Attributes:
        video_delta_indices: Indices of video observations to stack
        state_delta_indices: Indices of state observations to stack
        n_action_steps: Number of action steps to execute
        max_episode_steps: Maximum number of steps per episode
    """

    video_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    state_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    n_action_steps: int = 16
    max_episode_steps: int = 720
    terminate_on_success: bool = False


@dataclass
class WrapperConfigs:
    """Container for various environment wrapper configurations.

    Attributes:
        video: Configuration for video recording
        multistep: Configuration for multi-step processing
    """

    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)


def get_simpler_env_fn(
    env_name: str,
):
    def env_fn():
        from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs

        register_simpler_envs()
        return gym.make(env_name)

    return env_fn


def get_libero_env_fn(
    env_name: str,
):
    def env_fn():
        from gr00t.eval.sim.LIBERO_plus.libero_plus_env_seg import register_libero_plus_envs_seg

        register_libero_plus_envs_seg()
        return gym.make(env_name)

    return env_fn

def get_robocasa_env_fn(
    env_name: str,
):
    def env_fn():
        if env_name.startswith("robocasa365_panda_omron/"):
            import gr00t.eval.sim.robocasa365.gymnasium_groot  # noqa: F401
        else:
            import robocasa  # noqa: F401
            import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401

        return gym.make(env_name, enable_render=True)

    return env_fn


def get_gym_env(env_name: str, env_idx: int, total_n_envs: int):
    """Create Ray environment factory function without wrappers."""

    env_embodiment = get_embodiment_tag_from_env_name(env_name)
    env_prefix = env_name.split("/")[0]

    if env_prefix in ("robocasa_panda_omron", "robocasa365_panda_omron", "gr1_unified"):
        env_fn = get_robocasa_env_fn(env_name)

    elif env_embodiment in (EmbodimentTag.SIMPLER_ENV_GOOGLE, EmbodimentTag.SIMPLER_ENV_WIDOWX):
        env_fn = get_simpler_env_fn(env_name)

    elif env_embodiment in (EmbodimentTag.LIBERO_PANDA,):
        env_fn = get_libero_env_fn(env_name)

    else:
        raise ValueError(f"Invalid environment name: {env_name}")

    return env_fn()


def create_eval_env(
    env_name: str, env_idx: int, total_n_envs: int, wrapper_configs: WrapperConfigs
) -> gym.Env:
    """Create a single evaluation environment with wrappers.

    Args:
        env_name: Name of the gymnasium environment to use
        idx: Environment index (used to determine video recording)
        wrapper_configs: Configuration for environment wrappers
    Returns:
        Wrapped gymnasium environment
    """

    env = get_gym_env(env_name, env_idx, total_n_envs)
    if wrapper_configs.video.video_dir is not None:
        from gr00t.eval.sim.wrapper.video_recording_wrapper import (
            VideoRecorder,
            VideoRecordingWrapper,
        )

        record_video_keys = wrapper_configs.video.record_video_keys
        if record_video_keys is None and env_name.split("/")[0] in (
            "robocasa_panda_omron",
            "robocasa365_panda_omron",
        ):
            record_video_keys = ROBOCASA_PANDA_RECORD_VIDEO_KEYS

        video_recorder = VideoRecorder.create_h264(
            fps=wrapper_configs.video.fps,
            codec=wrapper_configs.video.codec,
            input_pix_fmt=wrapper_configs.video.input_pix_fmt,
            crf=wrapper_configs.video.crf,
            thread_type=wrapper_configs.video.thread_type,
            thread_count=wrapper_configs.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=Path(wrapper_configs.video.video_dir),
            steps_per_render=wrapper_configs.video.steps_per_render,
            max_episode_steps=wrapper_configs.video.max_episode_steps,
            overlay_text=wrapper_configs.video.overlay_text,
            record_video_keys=record_video_keys,
        )

    env = MultiStepWrapper(
        env,
        video_delta_indices=wrapper_configs.multistep.video_delta_indices,
        state_delta_indices=wrapper_configs.multistep.state_delta_indices,
        n_action_steps=wrapper_configs.multistep.n_action_steps,
        max_episode_steps=wrapper_configs.multistep.max_episode_steps,
        terminate_on_success=wrapper_configs.multistep.terminate_on_success,
    )
    return env


class _RobustAsyncVectorEnv(gym.vector.AsyncVectorEnv):
    """AsyncVectorEnv that tolerates variable-shaped info arrays across envs.

    Gymnasium's default _add_info pre-allocates a numpy array based on the
    first env's value shape and then assigns subsequent envs into it.  When
    envs return differently-shaped values (e.g. variable-length contact arrays)
    the assignment raises ValueError.  We catch that and fall back to a plain
    Python list for that key so the rest of the step can proceed normally.
    """

    def _add_info(self, infos, info, env_num):
        for k, v in info.items():
            if k not in infos:
                infos[k] = [None] * self.num_envs
                infos[f"_{k}"] = np.zeros(self.num_envs, dtype=bool)
            if isinstance(infos[k], np.ndarray):
                try:
                    infos[k][env_num] = v
                except (ValueError, TypeError):
                    lst = list(infos[k])
                    lst[env_num] = v
                    infos[k] = lst
            else:
                infos[k][env_num] = v
            infos[f"_{k}"][env_num] = True
        return infos

import numpy as np
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image, ImageDraw
from matplotlib import colormaps

IMAGE_TOKEN_ID = 151655


# def _normalize(x, percentile=95, gamma=4.0):
#     x = x.astype(np.float32)
#     x = x - x.min()

#     if x.max() < 1e-8:
#         return np.zeros_like(x)

#     vmax = np.percentile(x, percentile)

#     if vmax < 1e-8:
#         vmax = x.max()

#     x = np.clip(x, 0, vmax)
#     x = x / (vmax + 1e-8)
#     x = np.power(x, gamma)

#     return x



def _normalize(x):
    x = x.astype(np.float32)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    x = np.sqrt(x)
    return x


def resize_nearest(x, output_size):
    return np.asarray(Image.fromarray(x.astype(np.float32), mode="F").resize(output_size, Image.Resampling.BILINEAR))


def apply_jet_colormap(heatmap):
    heatmap = np.clip(heatmap, 0, 1)
    return (255 * colormaps["jet"](heatmap)[..., :3]).astype(np.uint8)


def overlay_heatmap(rgb, heatmap, alpha=0.35, overlay=True):
    heatmap_color = apply_jet_colormap(heatmap)
    if not overlay:
        return heatmap_color
    return np.clip((1.0 - alpha) * rgb.astype(np.float32) + alpha * heatmap_color.astype(np.float32), 0, 255).astype(np.uint8)


def extract_two_view_heatmaps(token_scores, input_ids, output_size=(256, 256)):
    image_mask = input_ids == IMAGE_TOKEN_ID

    image_scores = token_scores[image_mask]

    num_img_tokens = len(image_scores)

    assert num_img_tokens == 128, f"No image token found for IMAGE_TOKEN_ID={IMAGE_TOKEN_ID}"
    assert num_img_tokens % 2 == 0, f"Expected even image token count, got {num_img_tokens}"

    tokens_per_view = num_img_tokens // 2
    grid_size = int(np.sqrt(tokens_per_view))

    assert grid_size * grid_size == tokens_per_view, f"tokens_per_view={tokens_per_view} is not square"

    front = image_scores[:tokens_per_view].reshape(grid_size, grid_size)
    wrist = image_scores[tokens_per_view:].reshape(grid_size, grid_size)

    front = resize_nearest(_normalize(front), output_size)
    wrist = resize_nearest(_normalize(wrist), output_size)

    return front, wrist


def put_text(frame, text, xy):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.text(xy, text, fill=(255, 255, 255))
    return np.asarray(img)


def build_compare_frame(
    sensitivity_scores,
    importance_scores,
    input_ids,
    rgb_front,
    rgb_wrist,
    env_step,
    denoise_step,
):
    sens_front, sens_wrist = extract_two_view_heatmaps(sensitivity_scores, input_ids)
    imp_front, imp_wrist = extract_two_view_heatmaps(importance_scores, input_ids)

    overlay = True

    sens_front_overlay = overlay_heatmap(rgb_front, sens_front, alpha=0.35, overlay=overlay)
    sens_wrist_overlay = overlay_heatmap(rgb_wrist, sens_wrist, alpha=0.35, overlay=overlay)

    imp_front_overlay = overlay_heatmap(rgb_front, imp_front, alpha=0.35, overlay=overlay)
    imp_wrist_overlay = overlay_heatmap(rgb_wrist, imp_wrist, alpha=0.35, overlay=overlay)

    row1 = np.concatenate([sens_front_overlay, imp_front_overlay], axis=1)
    row2 = np.concatenate([sens_wrist_overlay, imp_wrist_overlay], axis=1)

    frame = np.concatenate([row1, row2], axis=0)

    frame = put_text(frame, f"env={env_step} denoise={denoise_step}", (10, 25))
    frame = put_text(frame, "Sensitivity Front", (10, 55))
    frame = put_text(frame, "Token Importance Front", (280, 55))
    frame = put_text(frame, "Sensitivity Wrist", (10, 310))
    frame = put_text(frame, "Token Importance Wrist", (280, 310))

    return frame


def save_cam_video(ep_cam_data, save_dir, episode_id):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    assert len(ep_cam_data) > 0

    num_denoise_steps = len(ep_cam_data[0]["cam_data"])

    # Hardcoded switch: if True only log the last denoise step, else log all steps.
    only_last_denoise_step = True

    last_denoise_step = ep_cam_data[0]["cam_data"][-1]["denoise_step"]

    compare_writers = {}

    fps = 4

    for denoise_step in range(num_denoise_steps):

        if only_last_denoise_step and denoise_step != last_denoise_step:
            continue

        compare_path = save_dir / f"episode_{episode_id:04d}_compare_denoise_{denoise_step}.mp4"
        if compare_path.exists():
            compare_path.unlink()

        compare_writers[denoise_step] = imageio.get_writer(str(compare_path), fps=fps, codec="libx264", format="FFMPEG")
    try:
        for env_step, step_data in enumerate(ep_cam_data):
            ids = step_data["input_ids"]

            if ids.ndim == 2:
                assert ids.shape[0] == 1
                ids = ids[0]

            ids = np.asarray(ids)

            rgb_front = step_data["image"]
            rgb_wrist = step_data["wrist_image"]
            # breakpoint()

            while rgb_front.ndim > 3:
                rgb_front = rgb_front[0]

            while rgb_wrist.ndim > 3:
                rgb_wrist = rgb_wrist[0]


            rgb_front = rgb_front.astype(np.uint8)
            rgb_wrist = rgb_wrist.astype(np.uint8)

            cam_data = step_data["cam_data"]

            assert len(cam_data) == num_denoise_steps

            for denoise_data in cam_data:
                denoise_step = denoise_data["denoise_step"]

                if denoise_step not in compare_writers:
                    continue

                frame = build_compare_frame(
                    denoise_data["sensitivity"],
                    denoise_data["token_importance"],
                    ids,
                    rgb_front,
                    rgb_wrist,
                    env_step,
                    denoise_step,
                )

                compare_writers[denoise_step].append_data(frame)

    finally:
        for writer in compare_writers.values():
            writer.close()

    print(f"[CAM] saved episode {episode_id} with {num_denoise_steps} denoise videos")
    
# def save_cam_video(ep_cam_data, save_dir, episode_id):


#     print(len(ep_cam_data)) # 35
#     print(ep_cam_data[0].keys()) # dict_keys(['input_ids', 'cam_data'])
#     print(len(ep_cam_data[0]["cam_data"])) # 4 denoising steps
#     print(ep_cam_data[0]["cam_data"][0].keys()) # dict_keys(['denoise_step', 't_discretized', 'sensitivity', 'token_importance', 'pred_velocity'])
#     sample = ep_cam_data[0]["cam_data"][0]
#     for key in sample.keys():
#         if isinstance(sample[key], np.ndarray):
#             print(key, sample[key].shape)
#     '''
#     sensitivity (149,)
#     token_importance (149,)
#     pred_velocity (40, 132)
#     '''
#     input_ids = ep_cam_data[0]["input_ids"]
#     # print(input_ids.shape) # (1,149)
#     '''
#     input_ids array([[151644,    872,    198, 151652, 151655, 151655, 151655, 151655,
#             151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
#             151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
#             151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
#             151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
#             151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
#             151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
#             151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
#             151655, 151655, 151655, 151655, 151653, 151652, 151655, 151655,
#             151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
#             151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
#             151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
#             151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
#             151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
#             151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
#             151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
#             151655, 151655, 151655, 151655, 151655, 151655, 151653,    628,
#             2176,    279,  27790,  19174,    323,    279,  41020,  19187,
#             304,    279,  14024, 151645,    198]])
#     '''
#     input_ids = input_ids[0]
#     image_mask = (input_ids == 151655)

#     image_token_indices = np.where(input_ids == 151655)[0]
#     print(image_token_indices)


def run_rollout_gymnasium_policy(
    env_name: str,
    policy: BasePolicy,
    wrapper_configs: WrapperConfigs,
    n_episodes: int = 10,
    n_envs: int = 1,
    seed: int | None = None,
    args: None = None,
) -> Any:
    """Run policy rollouts in parallel environments.

    Args:
        env_name: Name of the gymnasium environment to use
        policy: Policy instance
        n_episodes: Number of episodes to run
        n_envs: Number of parallel environments
        wrapper_configs: Configuration for environment wrappers
        seed: If set, forwards per-env seeds (``seed+i``) to the first
            ``env.reset`` so each sub-env is reproducible. Should be paired
            with :func:`gr00t.utils.determinism.seed_everything` upstream to
            also constrain policy-side RNGs.
    Returns:
        Collection results from running the episodes
    """
    start_time = time.time()
    n_episodes = max(n_episodes, n_envs)
    print(f"Running collecting {n_episodes} episodes for {env_name} with {n_envs} vec envs")

    env_fns = [
        partial(
            create_eval_env,
            env_idx=idx,
            env_name=env_name,
            total_n_envs=n_envs,
            wrapper_configs=wrapper_configs,
        )
        for idx in range(n_envs)
    ]

    if n_envs == 1:
        env = gym.vector.SyncVectorEnv(env_fns)
    else:
        env = _RobustAsyncVectorEnv(
            env_fns,
            shared_memory=False,
            context="spawn",
        )

    # Storage for results
    episode_lengths: list[int] = []
    episode_rewards: list[float] = []
    current_rewards = [0.0] * n_envs
    current_lengths = [0] * n_envs
    completed_episodes = 0
    current_successes = [False] * n_envs
    episode_successes = []
    episode_infos = defaultdict(list)

    # Initial reset; if a seed is provided, give each sub-env a distinct but
    # deterministic seed so that parallel workers don't all start from the
    # same initial state while still being run-to-run reproducible.
    if seed is not None:
        reset_seeds = [int(seed) + i for i in range(n_envs)]
        observations, _ = env.reset(seed=reset_seeds)
    else:
        observations, _ = env.reset()
    policy.reset()
    i = 0
    pbar = tqdm(total=n_episodes, desc="Episodes")
    episode_cam_data = []

    num_step_tt_in_traj = args.num_step_tt_in_traj
    while completed_episodes < n_episodes:
        num_step_tt_in_traj = num_step_tt_in_traj - 1
        options = {
            'tt_update': args.tt_update,
            'num_step_tt_in_traj': num_step_tt_in_traj
        }
        actions, info = policy.get_action(observations, options=options)
        # cam_data = info['cam_data']
        # len(cam_data) = 4
        # cam_data[0].keys() dict_keys(['denoise_step', 't_discretized', 'sensitivity', 'token_importance', 'pred_velocity'])
        input_ids = info["input_ids"]
        cam_data = info["cam_data"]
        episode_cam_data.append({"input_ids": input_ids, "cam_data": cam_data, 'image': observations['video.image'], 'wrist_image': observations['video.wrist_image']})

        next_obs, rewards, terminations, truncations, env_infos = env.step(actions)
        # NOTE (FY): Currently we don't properly handle policy reset. For now, our policy are stateless,
        # but in the future if we need policy to be stateful, we need to detect env reset and call policy.reset()
        i += 1
        # Update episode tracking
        for env_idx in range(n_envs):
            if "success" in env_infos:
                env_success = env_infos["success"][env_idx]
                if isinstance(env_success, list):
                    env_success = np.any(env_success)
                elif isinstance(env_success, np.ndarray):
                    env_success = np.any(env_success)
                elif isinstance(env_success, bool):
                    env_success = env_success
                elif isinstance(env_success, int):
                    env_success = bool(env_success)
                else:
                    raise ValueError(f"Unknown success dtype: {type(env_success)}")
                current_successes[env_idx] |= bool(env_success)
            else:
                current_successes[env_idx] = False

            if "final_info" in env_infos and env_infos["final_info"][env_idx] is not None:
                env_success = env_infos["final_info"][env_idx]["success"]
                if isinstance(env_success, list):
                    env_success = any(env_success)
                elif isinstance(env_success, np.ndarray):
                    env_success = np.any(env_success)
                elif isinstance(env_success, bool):
                    env_success = env_success
                elif isinstance(env_success, int):
                    env_success = bool(env_success)
                else:
                    raise ValueError(f"Unknown success dtype: {type(env_success)}")
                current_successes[env_idx] |= bool(env_success)
            current_rewards[env_idx] += rewards[env_idx]
            current_lengths[env_idx] += 1

            # If episode ended, store results
            if terminations[env_idx] or truncations[env_idx]:
                if "final_info" in env_infos:
                    current_successes[env_idx] |= any(env_infos["final_info"][env_idx]["success"])
                if "task_progress" in env_infos:
                    episode_infos["task_progress"].append(env_infos["task_progress"][env_idx][-1])
                if "q_score" in env_infos:
                    episode_infos["q_score"].append(np.max(env_infos["q_score"][env_idx]))
                if "valid" in env_infos:
                    episode_infos["valid"].append(all(env_infos["valid"][env_idx]))
                # Accumulate per-episode results. Both lists are captured
                # BEFORE the per-env trackers are reset to 0 below — without
                # this ordering downstream consumers silently see
                # episode_length=0 / episode_reward=0.0.
                episode_lengths.append(current_lengths[env_idx])
                episode_rewards.append(float(current_rewards[env_idx]))
                episode_successes.append(current_successes[env_idx])
                # Reset trackers for this environment.
                current_successes[env_idx] = False
                # only update completed_episodes if valid
                if "valid" in episode_infos:
                    if episode_infos["valid"][-1]:
                        completed_episodes += 1
                        pbar.update(1)
                else:
                    # envs don't return valid
                    completed_episodes += 1
                    pbar.update(1)
                # Reset with `0.0` to match the `[0.0] * n_envs` init and the
                # `float(...)` cast on line 347; otherwise the per-env entry's
                # static type silently flips int <-> float across iterations.
                current_rewards[env_idx] = 0.0
                current_lengths[env_idx] = 0
            
                ### Save cam
                save_cam_video(
                    episode_cam_data,
                    save_dir=args.save_cam_video_dir,
                    episode_id=completed_episodes,
                )
                episode_cam_data = []

                ### Reset policy
                policy.reset()
                num_step_tt_in_traj = args.num_step_tt_in_traj

        observations = next_obs
    pbar.close()

    env.reset()
    env.close()
    print(f"Collecting {n_episodes} episodes took {time.time() - start_time} seconds")

    assert len(episode_successes) >= n_episodes, (
        f"Expected at least {n_episodes} episodes, got {len(episode_successes)}"
    )

    # `current_lengths[env_idx] += 1` runs before each termination check
    # above, so any captured episode must have stepped at least once.
    assert all(length >= 1 for length in episode_lengths), (
        f"Internal invariant violated: rollout produced zero-length episode(s) "
        f"in {episode_lengths!r}."
    )

    # Surface the per-episode length and reward that were tracked locally so
    # downstream metrics (SimplerEnv / LIBERO / Robocasa / Wholebody) can
    # read them off episode_infos instead of silently falling back to 0.
    # Planted BEFORE the "valid" filter so they get filtered in lockstep
    # with the other episode_infos fields.
    episode_infos["episode_lengths"] = episode_lengths
    episode_infos["episode_rewards"] = episode_rewards

    episode_infos = dict(episode_infos)  # Convert defaultdict to dict
    for key, value in episode_infos.items():
        assert len(value) == len(episode_successes), (
            f"Length of {key} is not equal to the number of episodes"
        )

    # process valid results
    if "valid" in episode_infos:
        valids = episode_infos["valid"]
        valid_idxs = np.where(valids)[0]
        episode_successes = [episode_successes[i] for i in valid_idxs]
        episode_infos = {k: [v[i] for i in valid_idxs] for k, v in episode_infos.items()}

    return env_name, episode_successes, episode_infos


def create_gr00t_sim_policy(
    model_path: str,
    embodiment_tag: EmbodimentTag,
    policy_client_host: str = "",
    policy_client_port: int | None = None,
    trt_engine_path: str = "",
    trt_mode: TrtMode = TrtMode.N17_FULL_PIPELINE,
    algo: str | None = None
) -> BasePolicy:
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    if policy_client_host and policy_client_port:
        from gr00t.policy.server_client import PolicyClient

        policy = PolicyClient(host=policy_client_host, port=policy_client_port)
    else:
        gr00t_policy = Gr00tPolicy(
            embodiment_tag=embodiment_tag,
            model_path=model_path,
            device=0,
        )
        if trt_engine_path:
            deploy_dir = str(Path(__file__).resolve().parents[2] / "scripts" / "deployment")
            if deploy_dir not in sys.path:
                sys.path.insert(0, deploy_dir)
            from trt_model_forward import setup_tensorrt_engines

            setup_tensorrt_engines(gr00t_policy, trt_engine_path, mode=trt_mode)
        policy = Gr00tSimPolicyWrapper(gr00t_policy)
    return policy


def run_gr00t_sim_policy(
    env_name: str,
    n_episodes: int,
    max_episode_steps: int,
    model_path: str = "",
    policy_client_host: str = "",
    policy_client_port: int | None = None,
    n_envs: int = 8,
    n_action_steps: int = 8,
    video_dir: str | None = None,
    trt_engine_path: str = "",
    trt_mode: TrtMode = TrtMode.N17_FULL_PIPELINE,
    seed: int | None = None,
    args: None = None,
):
    # seed_everything resolves `None` via the GR00T_EVAL_SEED env var and is a
    # no-op when that is also unset, so the historical non-deterministic
    # behavior is preserved by default. Returns the effective seed (or None)
    # which we forward to env.reset below.
    seed = seed_everything(seed)

    embodiment_tag = get_embodiment_tag_from_env_name(env_name)

    if video_dir is None:
        if model_path:
            parts = model_path.split("/")
            model_slug = parts[-3] if len(parts) >= 3 else parts[-1]
            video_dir = f"/tmp/sim_eval_videos_{model_slug}_ac{n_action_steps}_{uuid.uuid4()}"
        else:
            video_dir = f"/tmp/sim_eval_videos_{env_name}_ac{n_action_steps}_{uuid.uuid4()}"
    wrapper_configs = WrapperConfigs(
        video=VideoConfig(
            video_dir=video_dir,
            max_episode_steps=max_episode_steps,
        ),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
            terminate_on_success=True,
        ),
    )

    policy = create_gr00t_sim_policy(
        model_path,
        embodiment_tag,
        policy_client_host,
        policy_client_port,
        trt_engine_path=trt_engine_path,
        trt_mode=trt_mode,
        algo=args.algo
    )

    results = run_rollout_gymnasium_policy(
        env_name=env_name,
        policy=policy,
        wrapper_configs=wrapper_configs,
        n_episodes=n_episodes,
        n_envs=n_envs,
        seed=seed,
        args=args
    )
    print("Video saved to: ", wrapper_configs.video.video_dir)
    return results


@dataclass
class RolloutConfig:
    """Configuration for rollout policy evaluation."""

    max_episode_steps: int = 504
    """Maximum number of steps per episode."""

    n_episodes: int = 50
    """Number of episodes to run."""

    model_path: str = ""
    """Path to model checkpoint."""

    policy_client_host: str = ""
    """Host for policy client."""

    policy_client_port: int | None = None
    """Port for policy client."""

    env_name: str = "libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
    """Environment name."""

    n_envs: int = 8
    """Number of parallel environments."""

    n_action_steps: int = 8
    """Number of action steps."""

    video_dir: str | None = None
    """Directory to save videos. If None, uses /tmp/sim_eval_videos_<env>_<uuid>."""

    trt_engine_path: str = ""
    """Path to TRT engine directory. If set, uses TRT inference instead of PyTorch."""

    trt_mode: TrtMode = TrtMode.N17_FULL_PIPELINE
    """TRT mode: 'n17_full_pipeline' (all engines), 'vit_llm_only', or 'action_head'."""

    seed: int | None = None
    """Optional seed for deterministic evaluation. When set, seeds Python /
    NumPy / torch / cuda RNGs, enables cuDNN determinism, and forwards
    per-env seeds to the sim envs. If left as ``None``, falls back to the
    ``GR00T_EVAL_SEED`` env var; if that is also unset, keeps the historical
    non-deterministic behavior."""

    algo: str | None = None
    save_cam_video_dir: str | None = None
    tt_update: int | None = None
    num_step_tt_in_traj: int | None = None


if __name__ == "__main__":
    args = tyro.cli(RolloutConfig)

    # validate policy configuration
    assert (args.model_path and not (args.policy_client_host or args.policy_client_port)) or (
        not args.model_path and args.policy_client_host and args.policy_client_port is not None
    ), (
        "Invalid policy configuration: You must provide EITHER model_path OR (policy_client_host & policy_client_port), not both.\n"
        "If all 3 arguments are provided, explicitly choose one:\n"
        '  - To use policy client: set --policy-client-host and --policy-client-port, and set --model-path ""\n'
        '  - To use model path: set --model-path, and set --policy-client-host "" (and leave --policy-client-port unset)'
    )

    results = run_gr00t_sim_policy(
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        max_episode_steps=args.max_episode_steps,
        model_path=args.model_path,
        policy_client_host=args.policy_client_host,
        policy_client_port=args.policy_client_port,
        n_envs=args.n_envs,
        n_action_steps=args.n_action_steps,
        video_dir=args.video_dir,
        trt_engine_path=args.trt_engine_path,
        trt_mode=args.trt_mode,
        seed=args.seed,
        args=args
    )
    print("results: ", results)
    print("success rate: ", np.mean(results[1]))
