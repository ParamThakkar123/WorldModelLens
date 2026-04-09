"""Example 05: Belief Analysis - Surprise, concepts, saliency.

This example demonstrates:
1. Computing surprise timeline
2. Searching for concept alignment
3. Computing saliency maps
4. Detecting hallucinations
5. Visualizing belief analysis results
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
from world_model_lens.visualization import InterventionVisualizer


def main():
    print("=" * 60)
    print("World Model Lens - Belief Analysis Example")
    print("=" * 60)

    cfg = WorldModelConfig(d_h=128, n_cat=16, n_cls=16, d_action=4, d_obs=12288)
    adapter = DreamerV3Adapter(cfg)
    wm = HookedWorldModel(adapter=adapter, config=cfg)
    analyzer = BeliefAnalyzer(wm)

    T = 20
    n_cat, n_cls = cfg.n_cat, cfg.n_cls

    print("\n[1] Running forward pass...")

    obs_seq = torch.randn(T, 3, 64, 64)
    action_seq = torch.randn(T, cfg.d_action)

    traj, cache = wm.run_with_cache(obs_seq, action_seq)

    print("\n[2] Computing surprise timeline...")

    surprise_result = analyzer.surprise_timeline(cache)
    print(f"    Mean surprise: {surprise_result.mean_surprise:.4f}")
    print(f"    Max surprise at t={surprise_result.max_surprise_timestep}")
    print(f"    Peak count: {len(surprise_result.peaks)}")

    print("\n[3] Searching for concept alignment...")

    pos_t = [0, 1, 2, 3, 4]
    neg_t = [10, 11, 12, 13, 14]

    concept_result = analyzer.concept_search(
        concept_name="early_vs_late",
        positive_timesteps=pos_t,
        negative_timesteps=neg_t,
        cache=cache,
        component="z_posterior",
    )

    print(f"    Top dims: {concept_result.top_dims[:5]}")
    print(f"    Method: {concept_result.method}")

    print("\n[4] Computing saliency...")

    saliency_result = analyzer.latent_saliency(
        traj=traj,
        cache=cache,
        timestep=5,
        target="reward_pred",
    )

    print(f"    h_saliency shape: {saliency_result.h_saliency.shape}")
    print(f"    z_saliency shape: {saliency_result.z_saliency.shape}")

    print("\n[5] Detecting hallucinations...")

    imagined = wm.imagine(start_state=traj.states[0], horizon=20)

    hallucination_result = analyzer.detect_hallucinations(
        real_traj=traj,
        imagined_traj=imagined,
        method="latent_distance",
        threshold=0.5,
    )

    print(f"    Severity score: {hallucination_result.severity_score:.4f}")
    print(f"    Hallucination timesteps: {hallucination_result.hallucination_timesteps}")

    print("\n[6] Building visualization dashboard...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("WorldModelLens - Belief Analysis Dashboard", fontsize=14)

    # 1. Surprise timeline (computed manually: DreamerV3 stores z as 1D so
    # the library's KL cache key is never written; derive KL from z_posterior/z_prior)
    ax = axes[0, 0]
    kl_vals_timeline = []
    for t in range(T):
        post = cache["z_posterior", t].reshape(n_cat, n_cls)
        prior = cache["z_prior", t].reshape(n_cat, n_cls)
        p = post.softmax(dim=-1).clamp(min=1e-8)
        q = prior.softmax(dim=-1).clamp(min=1e-8)
        kl_vals_timeline.append((p * (p.log() - q.log())).sum().item())
    ax.plot(range(T), kl_vals_timeline, marker="o", color="tomato")
    if surprise_result.peaks:
        peak_ts = [p for p in surprise_result.peaks if p < T]
        if peak_ts:
            peak_kl = [kl_vals_timeline[t] for t in peak_ts]
            ax.scatter(peak_ts, peak_kl, color="navy", zorder=5, label="Peaks", s=60)
            ax.legend(fontsize=8)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("KL Divergence")
    ax.set_title("Surprise Timeline")

    # 2. Per-category surprise heatmap
    ax = axes[0, 1]
    kl_per_cat = []
    for t in range(T):
        post = cache["z_posterior", t].reshape(n_cat, n_cls)
        prior = cache["z_prior", t].reshape(n_cat, n_cls)
        p = post.softmax(dim=-1).clamp(min=1e-8)
        q = prior.softmax(dim=-1).clamp(min=1e-8)
        kl_cat = (p * (p.log() - q.log())).sum(dim=-1).detach().numpy()
        kl_per_cat.append(kl_cat)
    heatmap_matrix = np.stack(kl_per_cat, axis=0)  # [T, n_cat]
    im = ax.imshow(heatmap_matrix.T, aspect="auto", origin="lower", cmap="hot")
    plt.colorbar(im, ax=ax, label="KL per category")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Latent Category")
    ax.set_title("Per-Category Surprise Heatmap")

    # 3. Saliency 
    ax = axes[1, 0]
    h_sal = saliency_result.h_saliency.detach().numpy()
    z_sal = saliency_result.z_saliency.detach().numpy()
    top_h = np.argsort(np.abs(h_sal))[-16:][::-1]
    top_z = np.argsort(np.abs(z_sal))[-16:][::-1]
    x = np.arange(16)
    ax.bar(x - 0.2, np.abs(h_sal[top_h]), width=0.4, label="h saliency", color="steelblue")
    ax.bar(x + 0.2, np.abs(z_sal[top_z]), width=0.4, label="z saliency", color="darkorange")
    ax.set_xlabel("Top-16 dimensions")
    ax.set_ylabel("|Saliency|")
    ax.set_title("Saliency at t=5 (reward_pred target)")
    ax.legend(fontsize=8)

    # 4. Real vs imagined state divergence
    ax = axes[1, 1]
    iv = InterventionVisualizer(wm)
    divergence_curve = iv.divergence_curve(traj, imagined)
    ts = sorted(divergence_curve.keys())
    divs = [divergence_curve[t] for t in ts]
    ax.plot(ts, divs, marker="o", color="purple")
    if hallucination_result.hallucination_timesteps:
        for ht in hallucination_result.hallucination_timesteps:
            if ht < len(ts):
                ax.axvline(ht, color="red", linestyle="--", alpha=0.6, linewidth=1)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("State Divergence (MSE)")
    ax.set_title(f"Real vs Imagined Divergence\n(severity={hallucination_result.severity_score:.3f})")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "belief_analysis_dashboard.png", dpi=120, bbox_inches="tight")
    print("    Saved belief_analysis_dashboard.png")
    plt.show()

    print("\n" + "=" * 60)
    print("Belief analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
