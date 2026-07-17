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

"""
LIBERO environment

This file wraps the original LIBERO as a Gymnasium environment,
and registers it so that it can be instantiated via gym.make(...) and work
using our distributed evaluation.
"""

import math
import os
import re

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from libero.libero import benchmark


# os.environ.setdefault("MUJOCO_GL", "egl")
# os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

from libero.libero.envs import OffScreenRenderEnv, SegmentationRenderEnv
from libero.libero.utils import get_libero_path
import numpy as np


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action


def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action


class LiberoProEnv(gym.Env):
    """LanguageTable env."""

    def __init__(self, task_bddl_file: str, task_description: str):
        # `ignore_done=True`: outer `MultiStepWrapper` owns truncation; robosuite's
        # horizon-termination is redundant and conflicts with LIBERO's done-override.
        self._env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=256,
            camera_widths=256,
            ignore_done=True,
        )
        self._task_description = task_description
        # Convert Gym action space to Gymnasium.
        self.observation_space = gym.spaces.Dict(
            {
                "video.image": gym.spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),
                "video.wrist_image": gym.spaces.Box(
                    low=0, high=255, shape=(256, 256, 3), dtype=np.uint8
                ),
                "state.x": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                "state.y": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                "state.z": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                "state.roll": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                "state.pitch": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                "state.yaw": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                "state.gripper": gym.spaces.Box(low=-1, high=1, shape=(2,)),
                "annotation.human.action.task_description": gym.spaces.Text(max_length=512),
            }
        )
        self.action_space = spaces.Dict(
            {
                "action.x": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.y": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.z": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.roll": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.pitch": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.yaw": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.gripper": spaces.Box(low=-1, high=1, shape=(1,)),
            }
        )

    def close(self):
        self._env.close()

    def _process_observation(self, obs):
        xyz = obs["robot0_eef_pos"]
        rpy = quat2axisangle(obs["robot0_eef_quat"])
        gripper = obs["robot0_gripper_qpos"]
        new_obs = {
            "video.image": obs["agentview_image"][::-1, ::-1],
            "video.wrist_image": obs["robot0_eye_in_hand_image"][::-1, ::-1],
            "state.x": [xyz[0]],
            "state.y": [xyz[1]],
            "state.z": [xyz[2]],
            "state.roll": [rpy[0]],
            "state.pitch": [rpy[1]],
            "state.yaw": [rpy[2]],
            "state.gripper": gripper,
            "annotation.human.action.task_description": self._task_description,
        }
        return new_obs

    def reset(self, seed=None, options=None):
        if seed is not None:
            # OffScreenRenderEnv follows the robosuite API: .seed(int), not reset(seed=...).
            self._env.seed(int(seed))
        observation = self._env.reset()
        observation = self._process_observation(observation)
        info = {"success": self._env.check_success()}
        return observation, info

    def step(self, action):
        action_vector = np.concatenate(
            [
                action["action.x"],
                action["action.y"],
                action["action.z"],
                action["action.roll"],
                action["action.pitch"],
                action["action.yaw"],
                action["action.gripper"],
            ],
            axis=0,
        )
        action_vector = normalize_gripper_action(action_vector)
        action_vector = invert_gripper_action(action_vector)
        observation, reward, done, info = self._env.step(action_vector)
        observation = self._process_observation(observation)
        info["success"] = self._env.check_success()
        truncated = False
        return observation, reward, done, truncated, info


# Base LIBERO suites and the LIBERO-PRO perturbation variants that ship as
# pre-generated bddl directories / benchmark suites (`<base>_<perturbation>`).
# NOTE: "env" (environment replacement) is NOT pre-shipped by LIBERO-PRO; it
# must be generated first (see `generate_perturbed_bddls` below) before it can
# be registered here.
_LIBERO_PRO_BASE_SUITES = ("libero_10", "libero_spatial", "libero_object", "libero_goal")
_LIBERO_PRO_PERTURBATIONS = ("swap", "lan", "object", "task")


def _read_bddl_language(bddl_path: str, fallback: str) -> str:
    """Return the instruction from a bddl `(:language ...)` block.

    LIBERO derives `task.language` from the *filename*, but the `_lan` / `_task`
    perturbations only change the language inside the bddl (the filename stays
    identical to the base task). Reading it here is what makes the policy
    actually see the perturbed instruction.
    """
    try:
        with open(bddl_path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return fallback
    match = re.search(r"\(:language\s*(.*?)\)", content, flags=re.S)
    if match:
        lang = match.group(1).strip()
        if lang:
            return lang
    return fallback


def register_libero_pro_envs(perturbations: tuple[str, ...] = _LIBERO_PRO_PERTURBATIONS):
    """Register base LIBERO tasks plus their LIBERO-PRO perturbed variants.

    Env ids keep the ``libero_sim/`` prefix (so the embodiment-tag lookup in
    ``env_utils`` still resolves to LIBERO_PANDA). Perturbed variants get a
    ``__<perturbation>`` suffix on the task portion, e.g.
    ``libero_sim/open_the_middle_drawer_of_the_cabinet__swap``.
    """
    benchmark_dict = benchmark.get_benchmark_dict()
    bddl_root = get_libero_path("bddl_files")

    for base_suite in _LIBERO_PRO_BASE_SUITES:
        # (suite_name, id_suffix) — base first, then each perturbation variant.
        suites = [(base_suite, "")]
        suites += [(f"{base_suite}_{p}", f"__{p}") for p in perturbations]

        for suite_name, id_suffix in suites:
            if suite_name not in benchmark_dict:
                # Perturbation variant not registered / not generated yet.
                continue
            task_suite = benchmark_dict[suite_name]()
            for task_id in range(task_suite.get_num_tasks()):
                task = task_suite.get_task(task_id)
                task_bddl_file = os.path.join(
                    bddl_root, task.problem_folder, task.bddl_file
                )
                if not os.path.exists(task_bddl_file):
                    continue
                task_description = _read_bddl_language(task_bddl_file, task.language)
                register(
                    id=f"libero_sim/{task.name}{id_suffix}",
                    entry_point="gr00t.eval.sim.LIBERO_pro.libero_pro_env:LiberoProEnv",
                    kwargs={
                        "task_bddl_file": task_bddl_file,
                        "task_description": task_description,
                    },
                )


if __name__ == "__main__":
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = "libero_10"  # can also choose libero_spatial, libero_object, etc.
    task_suite = benchmark_dict[task_suite_name]()
    # for key in [
    #     "libero_10",
    #     "libero_spatial",
    #     "libero_object",
    #     "libero_goal",
    #     "libero_90",
    # ]:
    #     for task_name in benchmark_dict[key]().get_task_names():
    #         print(f"- {key}/{task_name}")

    # retrieve a specific task
    for task_id in range(10):
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_description = task.language
        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        print(
            f"[info] retrieving task {task_id} from suite {task_suite_name}, the "
            + f"language instruction is {task_description}, and the bddl file is {task_bddl_file}"
        )

        # step over the environment
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": 128,
            "camera_widths": 128,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(0)
        env.reset()
        init_states = task_suite.get_task_init_states(
            task_id
        )  # for benchmarking purpose, we fix the a set of initial states
        init_state_id = 0
        env.set_init_state(init_states[init_state_id])

        dummy_action = [0.0] * 7
        for step in range(1):
            obs, reward, done, info = env.step(dummy_action)
            # print("step", step, "obs", obs.keys())

        # agentview_image
        rgb_image = obs['agentview_image'] # 128,128,3 - uint8
        import matplotlib.pyplot as plt
        plt.imsave(f"debug/rgb_img_{task_id}.png", rgb_image[::-1])
        
        env.close()
