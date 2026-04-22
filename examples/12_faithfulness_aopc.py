"""Example 12: Faithfulness Analysis with AOPC - Evaluate latent dimension faithfulness.

This example demonstrates:
1. Running the AOPC metric on latent representations
2. Analyzing perturbation curves for different components
3. Comparing faithfulness across model components
4. Visualizing AOPC results
"""

import pathlib

import torch
import matplotlib.pyplot as plt

OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "assets" / "examples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from world_model_lens import HookedWorldModel, WorldModelConfig
from world_model_lens.analysis import FaithfulnessAnalyzer
from world_model_lens.backends.toy_video_model import create_toy_video_adapter


def main():
    print("=" * 60)
    print("World Model Lens - AOPC Faithfulness Analysis Example")
    print("=" * 60)

    # Create a toy video world model for demonstration
    config = type("Config", (), {"latent_dim": 32, "hidden_dim": 128})()
    adapter = create_toy_video_adapter(latent_dim=32, hidden_dim=128)
    wm = HookedWorldModel(adapter=adapter, config=config, name="toy_video")

    print("\n[1] Running forward pass to collect activations...")

    # Generate synthetic video sequence
    obs_seq = torch.randn(50, 3, 32, 32)  # 50 frames of 32x32 RGB video
    traj, cache = wm.run_with_cache(obs_seq)

    print(f"Trajectory length: {len(traj)}")
    print(f"Cache keys: {list(cache.keys())}")

    # Initialize faithfulness analyzer
    analyzer = FaithfulnessAnalyzer(wm)

    print("\n[2] Computing AOPC for z_posterior component...")

    # Use reconstruction as predictor (common for faithfulness evaluation)
    def predictor_fn(cache):
        return cache["reconstruction"]

    # Compute AOPC with top-10 ablation
    aopc_result = analyzer.aopc(
        observations=obs_seq,
        target_component="z_posterior",
        predictor_fn=predictor_fn,
        max_k=10,
        normalize=True,
    )

    print(f"AOPC Score: {aopc_result.aopc_score:.4f}")
    print(f"Number of perturbation points: {len(aopc_result.mses)}")
    print(f"K values: {aopc_result.k_values}")

    print("\n[3] Computing detailed perturbation curve...")

    # Get full perturbation curve
    perturb_results = analyzer.perturbation_curve(
        observations=obs_seq,
        target_component="z_posterior",
        predictor_fn=predictor_fn,
        k_values=[1, 2, 5, 10, 15],
    )

    print("Perturbation results:")
    for res in perturb_results:
        print(f"K={res.k}: MSE delta={res.mse_delta:.4f}")
    print("\n[4] Comparing faithfulness across components...")

    components = ["z_posterior", "h"]
    component_scores = {}

    for comp in components:
        try:
            result = analyzer.aopc(
                observations=obs_seq, target_component=comp, predictor_fn=predictor_fn, max_k=5
            )
            component_scores[comp] = result.aopc_score
            print(f"{comp}: AOPC={result.aopc_score:.4f}")
        except Exception as e:
            print(f"Failed to compute AOPC for {comp}: {e}")

    print("\n[5] Visualizing AOPC curve...")

    # Create visualization
    fig = aopc_result.plot(figsize=(10, 6))
    plt.savefig(OUTPUT_DIR / "aopc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"AOPC curve saved to: {OUTPUT_DIR / 'aopc_curve.png'}")

    print("\n[6] Summary")
    print("=" * 40)
    print(f"AOPC Score: {aopc_result.aopc_score:.4f}")
    print(f"Components analyzed: {list(component_scores.keys())}")
    print(f"Component faithfulness scores: {component_scores}")

    # Find most faithful component
    if component_scores:
        most_faithful = max(component_scores, key=component_scores.get)
        print(
            f"Most faithful component: {most_faithful} (score: {component_scores[most_faithful]:.4f})"
        )

    print("\nExample completed! Higher AOPC scores indicate more faithful representations.")
    print("The AOPC metric helps identify which latent dimensions are most critical")
    print("for the model's predictive performance.")


if __name__ == "__main__":
    main()
