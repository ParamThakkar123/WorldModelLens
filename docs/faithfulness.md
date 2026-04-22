# Faithfulness Analysis

Faithfulness metrics evaluate how well model components (especially latent representations) explain or predict the model's behavior. This is crucial for interpretability, safety auditing, and understanding what information is encoded in different parts of the model.

## Area Over Perturbation Curve (AOPC)

The AOPC metric measures the faithfulness of latent dimensions by quantifying how much ablating (removing) the top-K most important dimensions affects model predictions.

### Concept

1. **Importance Ranking**: Dimensions are ranked by importance (default: magnitude of activations)
2. **Progressive Ablation**: For K = 1, 2, ..., N dimensions:
   - Zero out the top-K dimensions using hooks
   - Measure the change in model output (MSE delta)
3. **Integration**: Compute the area under the perturbation curve using trapezoidal integration

Higher AOPC scores indicate more faithful representations – the dimensions are critical for predictions.

### Mathematical Formulation

Given a component \( c \) with activations \( \mathbf{z} \in \mathbb{R}^{T \times D} \), predictor output \( p(\mathbf{z}) \), and importance scores \( s_d \) for each dimension \( d \):

1. Sort dimensions: \( d_1, d_2, \dots, d_D \) where \( s_{d_1} \geq s_{d_2} \geq \dots \geq s_{d_D} \)
2. For each K: Ablate top-K dimensions, compute MSE: \( \delta_K = \|p(\mathbf{z}) - p(\mathbf{z}_{\setminus \{d_1,\dots,d_K\}})\|^2 \)
3. AOPC = \( \int_0^D \delta(k) \, dk \) (trapezoidal approximation)

### Usage

```python
from world_model_lens.analysis import FaithfulnessAnalyzer

analyzer = FaithfulnessAnalyzer(wm)

# Basic AOPC computation
result = analyzer.aopc(observations, target_component="z_posterior")
print(f"AOPC Score: {result.aopc_score:.4f}")

# Detailed perturbation curve
curve = analyzer.perturbation_curve(observations, k_values=[1, 5, 10, 20])
for point in curve:
    print(f"K={point.k}: MSE delta={point.mse_delta:.4f}")
```

### Parameters

- `target_component`: Component to analyze (e.g., "z_posterior", "h")
- `predictor_fn`: Custom function to compute predictions from cache
- `max_k`: Maximum dimensions to ablate
- `normalize`: Normalize by max MSE for comparable scores
- `dim_importance`: Custom importance scores per dimension

## Related Concepts

### Attribution Faithfulness

AOPC is part of a broader class of attribution faithfulness metrics that evaluate how well explanation methods (saliency maps, feature attributions) reflect actual model behavior.

### Causal Faithfulness

In causal interpretability, faithfulness measures how well interventions on explanations correspond to interventions on the actual model.

### Information Faithfulness

Quantifies how much predictive information is preserved when projecting to lower-dimensional subspaces.

## Applications

### Model Debugging

Identify which latent dimensions are most critical for predictions, helping debug why certain inputs lead to specific outputs.

### Safety Auditing

Ensure that safety-critical information is encoded in interpretable dimensions that can be monitored or intervened upon.

### Architecture Design

Compare faithfulness across different model components to understand information flow and representation quality.

### Benchmarking

Standardized metric for comparing representation quality across different world model architectures.

## Implementation Details

- Uses activation hooks to zero out dimensions during forward pass
- Supports custom predictors for different output modalities
- Handles batched inputs and multi-timestep sequences
- Provides visualization methods for perturbation curves

## Limitations

- Computational cost scales with number of dimensions
- Assumes linear importance ranking (can be customized)
- Zero ablation may not reflect realistic interventions
- Sensitive to predictor choice and normalization

## References

- Samek et al. "Evaluating the Visualization of What a Deep Neural Network Has Learned" (2017)
- Montavon et al. "Methods for Interpreting and Understanding Deep Neural Networks" (2018)
- Adebayo et al. "Sanity Checks for Saliency Maps" (2018)