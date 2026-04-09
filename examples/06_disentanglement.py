"""Example 06: Disentanglement Analysis.

This example demonstrates:
1. Computing disentanglement metrics
2. Analyzing factor representations
3. Visualizing factor-dimension assignments
"""

import pathlib

import torch
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = pathlib.Path("assets/examples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from world_model_lens import HookedWorldModel, WorldModelConfig
from world_model_lens.backends.dreamerv3 import DreamerV3Adapter
from world_model_lens.analysis.belief_analyzer import BeliefAnalyzer
from world_model_lens.visualization import CacheSignalPlotter


def main():
    print("=" * 60)
    print("World Model Lens - Disentanglement Analysis Example")
    print("=" * 60)

    cfg = WorldModelConfig(d_h=128, n_cat=16, n_cls=16, d_action=4, d_obs=12288)
    adapter = DreamerV3Adapter(cfg)
    wm = HookedWorldModel(adapter=adapter, config=cfg)
    analyzer = BeliefAnalyzer(wm)

    print("\n[1] Collecting activations...")

    obs_seq = torch.randn(50, 3, 64, 64)
    action_seq = torch.randn(50, cfg.d_action)

    traj, cache = wm.run_with_cache(obs_seq, action_seq)

    print("\n[2] Creating synthetic factors...")

    factors = {
        "speed": torch.tensor([float(i % 10) / 10 for i in range(50)]),
        "direction": torch.tensor([float((i * 7) % 10) / 10 for i in range(50)]),
        "reward_level": torch.tensor([1.0 if i < 25 else 0.0 for i in range(50)]),
    }

    print(f"    Factors: {list(factors.keys())}")

    print("\n[3] Computing disentanglement metrics...")

    disentanglement_result = analyzer.disentanglement_score(
        cache=cache,
        factors=factors,
        metrics=["MIG", "DCI", "SAP"],
        component="z_posterior",
    )

    print(f"    MIG score: {disentanglement_result.scores.get('MIG', 0):.4f}")
    print(f"    DCI score: {disentanglement_result.scores.get('DCI', 0):.4f}")
    print(f"    SAP score: {disentanglement_result.scores.get('SAP', 0):.4f}")
    print(f"    Total score: {disentanglement_result.total_score:.4f}")

    print("\n[4] Factor assignments:")
    for factor, dims in disentanglement_result.factor_dim_assignment.items():
        print(f"    {factor}: dims {dims[:5]}...")

    print("\n[5] Building visualization dashboard...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("WorldModelLens - Disentanglement Analysis Dashboard", fontsize=14)

    # 1. MIG/DCI/SAP scores
    ax = axes[0]
    metric_names = ["MIG", "DCI", "SAP"]
    metric_vals = [disentanglement_result.scores.get(m, 0.0) for m in metric_names]
    colors = ["steelblue", "darkorange", "mediumseagreen"]
    bars = ax.bar(metric_names, metric_vals, color=colors)
    ax.set_ylabel("Score")
    ax.set_title(f"Disentanglement Metrics\n(total={disentanglement_result.total_score:.4f})")
    for bar, val in zip(bars, metric_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 2. Factor-dimension assignment heatmap
    ax = axes[1]
    factor_names = list(disentanglement_result.factor_dim_assignment.keys())
    n_dims_show = 32
    heatmap_data = np.zeros((len(factor_names), n_dims_show))
    for i, fname in enumerate(factor_names):
        for d in disentanglement_result.factor_dim_assignment[fname]:
            if d < n_dims_show:
                heatmap_data[i, d] = 1.0
    if heatmap_data.sum() == 0:  # Fallback: show per-dim variance if no clear factor assignment
        z_seq = cache["z_posterior"]  # [50 timesteps, d_z]
        var_per_dim = z_seq.var(dim=0).detach().numpy()[:n_dims_show]
        heatmap_data = np.tile(var_per_dim, (len(factor_names), 1))
    im = ax.imshow(heatmap_data, aspect="auto", cmap="Blues")
    ax.set_yticks(range(len(factor_names)))
    ax.set_yticklabels(factor_names)
    ax.set_xlabel("Latent Dimension")
    ax.set_title("Factor-Dimension Assignment")
    plt.colorbar(im, ax=ax)

    # 3. PCA of z_posterior
    ax = axes[2]
    z_seq = cache["z_posterior"]  # [50 timesteps, d_z]
    z_centered = z_seq - z_seq.mean(dim=0)
    _, _, Vt = torch.pca_lowrank(z_centered, q=2)
    pca_proj = (z_centered @ Vt).detach().numpy()  # [50 timesteps, 2 PCA dims]

    reward_colors = factors["reward_level"].numpy()
    sc = ax.scatter(
        pca_proj[:, 0],
        pca_proj[:, 1],
        c=reward_colors,
        cmap="RdYlGn",
        s=30,
        alpha=0.8,
    )
    plt.colorbar(sc, ax=ax, label="reward_level")
    # Annotate 5 evenly spaced timesteps:
    # 0, 50//4=12, 50//2=25, 3*50//4=37, 50-1
    for i in [0, 12, 25, 37, 49]:
        ax.annotate(str(i), (pca_proj[i, 0], pca_proj[i, 1]), fontsize=7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("z_posterior PCA (colored by reward_level)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "disentanglement_dashboard.png", dpi=120, bbox_inches="tight")
    print("    Saved disentanglement_dashboard.png")
    plt.show()

    print("\n" + "=" * 60)
    print("Disentanglement analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
