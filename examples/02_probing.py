"""Example 02: Probing - Train linear probes on cached activations.

This example demonstrates:
1. Collecting activations from multiple trajectories
2. Creating synthetic labels for concepts
3. Training linear probes
4. Analyzing probe results
5. Visualizing latent structure
"""

import pathlib

import pathlib

import numpy as np
import torch
import matplotlib.pyplot as plt

OUTPUT_DIR = pathlib.Path("assets/examples")

OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "assets" / "examples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from world_model_lens import HookedWorldModel, LatentProber, WorldModelConfig
from world_model_lens.backends.dreamerv3 import DreamerV3Adapter
from world_model_lens.visualization import CacheSignalPlotter


def main():
    print("=" * 60)
    print("World Model Lens - Linear Probing Example")
    print("=" * 60)

    cfg = WorldModelConfig(d_h=128, n_cat=16, n_cls=16, d_action=4, d_obs=12288)
    adapter = DreamerV3Adapter(cfg)
    wm = HookedWorldModel(adapter=adapter, config=cfg)

    print("\n[1] Collecting activations from 5 trajectories...")

    all_activations = []
    all_labels = []

    for ep in range(5):
        obs_seq = torch.randn(20, 3, 64, 64)
        action_seq = torch.randn(20, cfg.d_action)

        traj, cache = wm.run_with_cache(obs_seq, action_seq)

        z_posterior = cache["z_posterior"]
        z_flat = z_posterior.flatten(1) if z_posterior.dim() > 2 else z_posterior
        all_activations.append(z_flat)

        labels = np.random.randint(0, 3, size=len(z_flat))
        all_labels.append(labels)

    activations = torch.cat(all_activations, dim=0)
    labels = np.concatenate(all_labels)

    print(f"    Collected {activations.shape[0]} activations, shape: {activations.shape}")

    print("\n[2] Training linear probes...")

    prober = LatentProber(seed=42)

    concepts = {
        "reward_region": (labels == 0).astype(np.float32),
        "novel_state": (labels == 1).astype(np.float32),
        "high_value": (labels == 2).astype(np.float32),
    }

    sweep_result = prober.sweep(
        cache={"z_posterior": activations},
        activation_names=["z_posterior"],
        labels_dict=concepts,
        probe_type="linear",
    )

    print("\n[3] Probe Results:")
    for key, result in sweep_result.results.items():
        print(f"    {key}: accuracy={result.accuracy:.3f}")

    print("\n[4] Building visualization dashboard...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("WorldModelLens - Linear Probing Dashboard", fontsize=14)

    # 1. PCA of collected activations
    acts_centered = activations - activations.mean(0)
    _, _, Vt = torch.pca_lowrank(acts_centered, q=2)
    pca_proj = (acts_centered @ Vt).detach().numpy()

    ax = axes[0, 0]
    sc = ax.scatter(pca_proj[:, 0], pca_proj[:, 1], c=labels, cmap="tab10", s=8, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Label (0=reward, 1=novel, 2=value)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("z_posterior PCA (colored by concept label)")

    # 2. Probe accuracy per concept
    ax = axes[0, 1]
    concept_names = list(concepts.keys())
    accuracies = [
        sweep_result.results.get(f"{c}_z_posterior", None) for c in concept_names
    ]
    acc_vals = [r.accuracy if r is not None else 0.0 for r in accuracies]
    bars = ax.bar(concept_names, acc_vals, color="steelblue")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Probe Accuracy per Concept")
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Chance (0.5)")
    ax.legend(fontsize=8)
    for bar, acc in zip(bars, acc_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 3. Hidden state norm timeline for last run
    ax = axes[1, 0]
    h_signal = CacheSignalPlotter.plot_cache_signal(cache, "h")
    ax.plot(h_signal["timesteps"], h_signal["norms"], marker="o", color="darkorange")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("||h_t||")
    ax.set_title("Hidden State Norms (last episode)")

    # 4. Label distribution
    ax = axes[1, 1]
    unique, counts = np.unique(labels, return_counts=True)
    label_names = {0: "reward_region", 1: "novel_state", 2: "high_value"}
    ax.bar([label_names.get(int(u), str(u)) for u in unique], counts, color="mediumseagreen")
    ax.set_ylabel("Count")
    ax.set_title("Label Distribution (across all episodes)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "probing_dashboard.png", dpi=120, bbox_inches="tight")
    print("    Saved probing_dashboard.png")
    plt.show()

    print("\n" + "=" * 60)
    print("Probing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
