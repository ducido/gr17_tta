"""Generate ISS heatmaps for one prepared X-ICM episode."""

from __future__ import annotations

import json

import hydra
from iss_nmr_toolkit.constants import DEFAULT_VIEWS
from iss_nmr_toolkit.core.iss import compute_iss
from iss_nmr_toolkit.io.artifacts import save_iss_npz
from iss_nmr_toolkit.io.episode import load_episode_directory
from iss_nmr_toolkit.runner_utils import (
    iss_path,
    output_dir,
    quiet_third_party_logs,
    resolve_episode_dir,
    resolve_prompt,
)
from omegaconf import DictConfig, OmegaConf

### for gr00t
import os
import torch
import gr00t

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
###

@hydra.main(version_base=None, config_path="vla_explain_configs", config_name="libero")
def main(cfg: DictConfig) -> None:
    quiet_third_party_logs()
    episode_dir = resolve_episode_dir(cfg)
    out_dir = output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = Gr00tPolicy(
        model_path=cfg.policy.checkpoint_dir,
        embodiment_tag=EmbodimentTag.resolve(cfg.policy.embodiment_tag),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        strict=True,
    )

    dataset = LeRobotEpisodeLoader(
        dataset_path=cfg.data.root,
        modality_configs=policy.get_modality_config(),
    )
    observations = dataset[cfg.data.episode]


    heatmaps = compute_iss(
        policy,
        observations,
        views=DEFAULT_VIEWS,
        n_masks=int(cfg.iss.n_masks),
        p_keep=float(cfg.iss.p_keep),
        grid_size=(int(cfg.iss.grid_size[0]), int(cfg.iss.grid_size[1])),
        blur_sigma=float(cfg.iss.blur_sigma),
        time_stride=int(cfg.iss.time_stride),
        batch_size=int(cfg.iss.batch_size),
        seed=int(cfg.iss.seed),
    )

    path = save_iss_npz(
        iss_path(cfg),
        heatmaps,
        meta={
            "episode_dir": str(episode_dir.resolve()),
            "policy": "pi05",
            "config_name": cfg.policy.config_name,
            "n_masks": int(cfg.iss.n_masks),
            "p_keep": float(cfg.iss.p_keep),
            "grid_size": [int(cfg.iss.grid_size[0]), int(cfg.iss.grid_size[1])],
            "blur_sigma": float(cfg.iss.blur_sigma),
            "time_stride": int(cfg.iss.time_stride),
            "seed": int(cfg.iss.seed),
            "views": DEFAULT_VIEWS,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
    )
    summary = {
        "iss_heatmaps": str(path),
        "episode_dir": str(episode_dir),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    with (out_dir / "iss_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved ISS heatmaps: {path}")


if __name__ == "__main__":
    main()
