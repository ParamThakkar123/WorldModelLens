# Examples

World Model Lens includes 10 comprehensive examples demonstrating core interpretability techniques, from quickstart to advanced causal analysis. All examples are self-contained and can be run independently.

## Running Examples

All examples are located in the `examples/` directory. Run any example with:

```bash
python examples/01_quickstart.py
python examples/02_probing.py
# ... etc
```

Or from the repo root:

```bash
cd examples
python 01_quickstart.py
```

## Example Categories

```{toctree}
:maxdepth: 2

examples/getting-started
examples/analysis-techniques
examples/advanced-analysis
examples/non-rl-models
examples/causal-analysis
```

---

## Quick Reference

| Example | Focus | Key Concept |
|---------|-------|-------------|
| 01 | Quickstart | Basic workflow: cache → probe |
| 02 | Linear Probing | Train probes on activations |
| 03 | Activation Patching | Causal intervention via patching |
| 04 | Imagination Branching | Fork & compare imagined futures |
| 05 | Belief Analysis | Surprise, saliency, concepts |
| 06 | Disentanglement | Analyze factor representations |
| 07 | Video Model | Non-RL video prediction model |
| 08 | Toy Video Analysis | Video model with geometry & memory |
| 09 | Toy Scientific Dynamics | Scientific model (Lorenz, pendulum) |
| 10 | Counterfactual Engine | Interventions & branch trees |

## What Each Example Teaches

### Core Workflow
**Example 01** shows the essential pattern used by all examples: create → run with cache → extract results.

### Interpretability Techniques
**Examples 02-06** cover the main interpretability methods:
- Probing: What do activations encode?
- Patching: What causes outputs?
- Branching: What if we changed this state?
- Analysis: When does the model surprise? What factors matter?

### Model Diversity
**Examples 07-09** prove World Model Lens works with **any** world model architecture:
- RL agents (DreamerV3)
- Video prediction (no actions, no rewards)
- Scientific dynamics (pure latent dynamics)

### Causal Reasoning
**Example 10** demonstrates advanced counterfactual analysis:
- Single interventions
- Branch trees
- Systematic comparison

## Prerequisites

All examples require:

```bash
pip install world_model_lens
```

Optional for visualization/analysis features:

```bash
pip install world_model_lens[viz,dev]
```

## Common Patterns

### Pattern 1: Forward Pass with Caching

```python
from world_model_lens import HookedWorldModel, WorldModelConfig
from world_model_lens.backends.dreamerv3 import DreamerV3Adapter

cfg = WorldModelConfig(...)
adapter = DreamerV3Adapter(cfg)
wm = HookedWorldModel(adapter=adapter, config=cfg)

# Run and collect all activations
traj, cache = wm.run_with_cache(obs_seq, action_seq)

# Access cached activations
h_t = cache["h", 0]  # Hidden state at t=0
z_posterior = cache["z_posterior", 0]
```

### Pattern 2: Imagination (Rollout)

```python
# Continue from a real state
imagined = wm.imagine(start_state=traj.states[5], horizon=20)

# Or with specific actions
actions = torch.randn(20, cfg.d_action)
imagined = wm.imagine(start_state=traj.states[5], actions=actions, horizon=20)
```

### Pattern 3: Analysis with Belief Analyzer

```python
from world_model_lens.analysis.belief_analyzer import BeliefAnalyzer

analyzer = BeliefAnalyzer(wm)

# Get surprise timeline
surprise = analyzer.surprise_timeline(cache)

# Find concepts
concepts = analyzer.concept_search(
    concept_name="early_vs_late",
    positive_timesteps=[0, 1, 2],
    negative_timesteps=[10, 11, 12],
    cache=cache
)
```

## Troubleshooting Examples

### Import Error: `ModuleNotFoundError: No module named 'world_model_lens'`

Ensure you've installed the package:

```bash
pip install -e .  # From repo root
# or
pip install world_model_lens  # From PyPI
```

### CUDA Out of Memory

Reduce tensor sizes in examples:

```python
# Instead of:
obs_seq = torch.randn(100, 3, 64, 64)
# Use:
obs_seq = torch.randn(20, 3, 64, 64)
```

### Print Output Too Verbose

Examples are designed to print progress. To suppress:

```python
import sys
sys.stdout = open('/dev/null', 'w')  # Linux/Mac
# Then run example...
```

Or just redirect stdout when running:

```bash
python examples/01_quickstart.py > /dev/null 2>&1
```

## Next Steps

After exploring these examples:

1. **Read the API docs** — Understand the full API surface
2. **Build your own** — Use these as templates for your models
3. **Integrate with your model** — Create a custom adapter for your world model
4. **Share results** — Contribute new examples or analysis tools!

See [Getting Started](getting-started.md) for a guided walkthrough.
