"""Visualize the distribution of timesteps produced by `sample_time`.

Mirrors `FlowmatchingActionHead.sample_time` in
gr00t/model/gr00t_n1d7/gr00t_n1d7.py:

    sample = Beta(noise_beta_alpha, noise_beta_beta).sample([batch_size])
    t = (1 - sample) * noise_s

Defaults are taken from the repo configs:
    noise_beta_alpha = 1.5   (gr00t/configs/finetune_config.py)
    noise_beta_beta  = 1.0
    noise_s          = 0.999 (gr00t/configs/model/gr00t_n1d7.py)

Usage:
    python scripts/visualize_sample_time.py
    python scripts/visualize_sample_time.py --alpha 1.5 --beta 1.0 --noise-s 0.999 \
        --num-samples 100000 --bins 100 --out sample_time_dist.png
"""

import argparse

import numpy as np
import torch
from torch.distributions import Beta


def sample_time(batch_size, alpha, beta, noise_s, device="cpu", dtype=torch.float32):
    """Standalone copy of FlowmatchingActionHead.sample_time."""
    beta_dist = Beta(torch.tensor(float(alpha)), torch.tensor(float(beta)))
    sample = beta_dist.sample([batch_size]).to(device, dtype=dtype)
    sample = (1 - sample) * noise_s
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alpha", type=float, default=1.0, help="Beta distribution alpha (noise_beta_alpha).")
    parser.add_argument("--beta", type=float, default=1.5, help="Beta distribution beta (noise_beta_beta).")
    parser.add_argument("--noise-s", type=float, default=0.999, help="Timestep scaling factor (noise_s).")
    parser.add_argument("--num-samples", type=int, default=100_000, help="Number of timesteps to sample.")
    parser.add_argument("--bins", type=int, default=100, help="Number of histogram bins.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--out", type=str, default="sample_time_dist.png", help="Output image path.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    samples = sample_time(args.num_samples, args.alpha, args.beta, args.noise_s).cpu().numpy()

    # In the flow-matching mix  noisy = (1 - t) * noise + t * actions:
    #   t -> 0  => pure noise      t -> 1  => clean action
    frac_noise_side = float((samples < 0.5).mean())
    frac_action_side = float((samples >= 0.5).mean())

    print(f"Beta(alpha={args.alpha}, beta={args.beta}), noise_s={args.noise_s}")
    print(f"n={args.num_samples}")
    print(f"  min    = {samples.min():.4f}")
    print(f"  max    = {samples.max():.4f}")
    print(f"  mean   = {samples.mean():.4f}")
    print(f"  median = {np.median(samples):.4f}")
    print(f"  std    = {samples.std():.4f}")
    print(f"  t < 0.5 (closer to PURE NOISE)   = {frac_noise_side:.1%}")
    print(f"  t >= 0.5 (closer to CLEAN ACTION) = {frac_action_side:.1%}")

    # Matplotlib is imported here so the sampling/stats still work without a display backend.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    counts, bin_edges, _ = ax.hist(
        samples, bins=args.bins, density=True, color="#4C72B0", alpha=0.75, edgecolor="white", linewidth=0.3
    )

    ax.axvline(samples.mean(), color="#C44E52", linestyle="--", linewidth=1.5, label=f"mean = {samples.mean():.3f}")
    ax.axvline(np.median(samples), color="#DD8452", linestyle=":", linewidth=1.5, label=f"median = {np.median(samples):.3f}")

    # Annotate the two ends: what the timestep means in the flow-matching mix.
    ymax = counts.max()
    ax.set_xlim(0, 1)
    ax.text(0.02, ymax * 0.95, "t → 0\npure noise", ha="left", va="top", fontsize=9, color="#555555")
    ax.text(0.98, ymax * 0.95, "t → 1\nclean action", ha="right", va="top", fontsize=9, color="#555555")

    ax.set_xlabel("timestep  t = (1 - Beta) * noise_s   (0 = pure noise, 1 = clean action)")
    ax.set_ylabel("density")
    ax.set_title(
        f"sample_time distribution\nBeta(α={args.alpha}, β={args.beta}), noise_s={args.noise_s}, n={args.num_samples:,}\n"
        f"{frac_noise_side:.0%} closer to noise  |  {frac_action_side:.0%} closer to clean action"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved histogram to {args.out}")


if __name__ == "__main__":
    main()
