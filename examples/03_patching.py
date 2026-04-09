"""Example 03: Activation Patching - Test causal roles of components.

This example demonstrates:
1. Setting up clean and corrupted runs
2. Patching specific components
3. Measuring recovery rate
4. Full patching sweep
5. Visualizing recovery heatmap and divergence
"""

import pathlib

import numpy as np
import torch
import matplotlib.pyplot as plt

OUTPUT_DIR = pathlib.Path("assets/examples")

from world_model_lens import HookContext, HookedWorldModel, HookPoint, WorldModelConfig
from world_model_lens.backends.dreamerv3 import DreamerV3Adapter
from world_model_lens.patching.patcher import TemporalPatcher
from world_model_lens.visualization import CacheSignalPlotter, InterventionVisualizer


def main():
    print("=" * 60)
    print("World Model Lens - Activation Patching Example")
    print("=" * 60)

    cfg = WorldModelConfig(d_h=128, n_cat=16, n_cls=16, d_action=4, d_obs=12288)
    adapter = DreamerV3Adapter(cfg)
    wm = HookedWorldModel(adapter=adapter, config=cfg)

    print("\n[1] Creating clean and corrupted runs...")

    obs_seq = torch.randn(15, 3, 64, 64)
    action_seq = torch.randn(15, cfg.d_action)

    clean_traj, clean_cache = wm.run_with_cache(obs_seq, action_seq)

    obs_corrupted = obs_seq.clone()
    obs_corrupted[5:] = torch.randn_like(obs_corrupted[5:])

    corrupted_traj, corrupted_cache = wm.run_with_cache(obs_corrupted, action_seq)

    print(f"    Clean cache timesteps: {len(clean_cache.timesteps)}")
    print(f"    Corrupted cache timesteps: {len(corrupted_cache.timesteps)}")

    print("\n[2] Running patching experiment...")

    patcher = TemporalPatcher(wm)

    components = ["h", "z_posterior", "z_prior"]
    timesteps = [5, 6, 7, 8, 9]

    def reward_metric(pred) -> float:
        return pred.mean().item() if pred is not None else 0.0

    sweep_result = patcher.full_sweep(
        clean_cache=clean_cache,
        corrupted_cache=corrupted_cache,
        components=components,
        metric_fn=reward_metric,
        t_range=timesteps,
        parallel=False,
        clean_obs_seq=obs_seq,
        clean_action_seq=action_seq,
    )

    print("\n[3] Top patches by recovery rate:")
    top_patches = sweep_result.top_k_patches(k=5)
    for patch in top_patches:
        print(f"    {patch.component}@t={patch.timestep}: recovery={patch.recovery_rate:.3f}")

    print("\n[4] Building visualization dashboard...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("WorldModelLens - Activation Patching Dashboard", fontsize=14)

    # 1. Recovery rate heatmap
    ax = axes[0]
    recovery_matrix = sweep_result.recovery_matrix().cpu().numpy()
    im = ax.imshow(recovery_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(sweep_result.components)))
    ax.set_yticklabels(sweep_result.components)
    ax.set_xticks(range(len(sweep_result.timesteps)))
    ax.set_xticklabels(sweep_result.timesteps)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Component")
    ax.set_title("Recovery Rate Heatmap")
    plt.colorbar(im, ax=ax, label="Recovery Rate")

    # 2. Divergence between clean and corrupted trajectories
    ax = axes[1]
    iv = InterventionVisualizer(wm)
    divergence_curve = iv.divergence_curve(clean_traj, corrupted_traj)
    ts = sorted(divergence_curve.keys())
    divs = [divergence_curve[t] for t in ts]
    cum_divs = []
    running = 0.0
    for d in divs:
        running += d
        cum_divs.append(running)
    ax.plot(ts, divs, marker="o", label="Step divergence", color="tomato")
    ax.plot(ts, cum_divs, marker="s", linestyle="--", label="Cumulative", color="navy")
    ax.axvline(5, color="gray", linestyle=":", linewidth=1.5, label="Corruption starts")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("MSE Divergence")
    ax.set_title("Clean vs Corrupted Divergence")
    ax.legend(fontsize=8)

    # 3. Hidden state norms: clean vs corrupted
    ax = axes[2]
    h_clean = CacheSignalPlotter.plot_cache_signal(clean_cache, "h")
    h_corrupt = CacheSignalPlotter.plot_cache_signal(corrupted_cache, "h")
    ax.plot(h_clean["timesteps"], h_clean["norms"], marker="o", label="Clean", color="steelblue")
    ax.plot(
        h_corrupt["timesteps"],
        h_corrupt["norms"],
        marker="s",
        linestyle="--",
        label="Corrupted",
        color="darkorange",
    )
    ax.axvline(5, color="gray", linestyle=":", linewidth=1.5, label="Corruption starts")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("||h_t||")
    ax.set_title("Hidden State Norms: Clean vs Corrupted")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "patching_dashboard.png", dpi=120, bbox_inches="tight")
    print("    Saved patching_dashboard.png")
    plt.show()

    print("\n" + "=" * 60)
    print("Patching complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
