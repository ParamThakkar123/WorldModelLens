# Causal Analysis with Counterfactual Engine

**File:** `examples/10_causal_engine.py`

**What it teaches:** How to perform systematic causal reasoning via the CounterfactualEngine—the most powerful tool in World Model Lens.

## Overview

The **CounterfactualEngine** builds on patching (Example 03) but provides:
- **Single interventions** with detailed metrics
- **Branch trees** — multiple counterfactuals from one fork point
- **Systematic comparison** — side-by-side metrics for all interventions
- **Divergence tracing** — how do counterfactuals drift over time?

## What Is a Counterfactual?

A counterfactual is: **"What would happen if we changed this?"**

Examples:
- "What if we zeroed latent dimension 3 at t=5?"
- "What if we removed all action input at t=7?"
- "What if we patched z_posterior from the clean run?"

The CounterfactualEngine lets you ask these questions systematically.

---

## Example Walkthrough

### Part 1: Setup

```python
from world_model_lens.causal import CounterfactualEngine, Intervention

cfg = WorldModelConfig(...)
adapter = DreamerV3Adapter(cfg)
wm = HookedWorldModel(adapter=adapter, config=cfg)

# Data
obs_seq = torch.randn(15, 3, 64, 64)
action_seq = torch.randn(15, cfg.d_action)

# Engine
engine = CounterfactualEngine(wm)

# Baseline
baseline_traj, cache = wm.run_with_cache(obs_seq, action_seq)
```

### Part 2: Single Intervention

Create one counterfactual by intervening at a specific point:

```python
from world_model_lens.causal import Intervention

# Define intervention: zero dimensions 0–2 of z_posterior at t=5
ablate_z = Intervention(
    target_timestep=5,
    target_type="dimension",
    target_indices=[0, 1, 2],
    description="Ablate z dims 0–2 at t=5",
)

# Run counterfactual
cf_traj = engine.intervene(
    observations=obs_seq,
    intervention=ablate_z,
    actions=action_seq,
)

# Compare outcomes
from world_model_lens.causal import rollout_comparison
metrics = rollout_comparison(baseline_traj, cf_traj)

for key, value in metrics.to_dict().items():
    print(f"{key}: {value:.6f}")
```

**Output:**
```
reward_divergence: 0.234567
state_divergence: 0.123456
action_divergence: 0.345678
value_divergence: 0.456789
```

### Part 3: Divergence Tracing

Track how counterfactuals diverge from baseline over time:

```python
div_curve = engine.trace_divergence(baseline_traj, cf_traj)

# Sample at specific timesteps
for t in [0, 7, 14]:
    print(f"t={t}: cumulative divergence = {div_curve[t]:.6f}")
```

**Interpretation:**
- Low divergence early → intervention effect takes time to propagate
- Divergence increasing → effect compounds
- Divergence plateauing → system recovers or stabilizes

### Part 4: Branch Tree

Compare multiple interventions from the same fork point:

```python
interventions = [
    Intervention(
        target_timestep=5,
        target_type="dimension",
        target_indices=[0],
        description="Ablate dim 0",
    ),
    Intervention(
        target_timestep=5,
        target_type="dimension",
        target_indices=[1],
        description="Ablate dim 1",
    ),
    Intervention(
        target_timestep=5,
        target_type="action",
        target_indices=None,
        description="Zero all actions at t=5",
    ),
]

tree = engine.build_branch_tree(
    observations=obs_seq,
    interventions=interventions,
    base_actions=action_seq,
)

print(f"Tree with {len(tree.branches)} branches:")
for i, branch in enumerate(tree.branches):
    print(f"  [{i}] {branch.intervention.description}")
    print(f"      Fork: t={branch.fork_point}")
    print(f"      Divergence: {branch.divergence:.6f}")
```

**Output:**
```
Tree with 3 branches:
  [0] Ablate dim 0
      Fork: t=5
      Divergence: 0.345678
  [1] Ablate dim 1
      Fork: t=5
      Divergence: 0.234567
  [2] Zero all actions at t=5
      Fork: t=5
      Divergence: 0.876543
```

### Part 5: Intervention Comparison

Tabular comparison of all interventions with consistent metrics:

```python
compared = engine.compare_interventions(
    observations=obs_seq,
    interventions=interventions,
    base_actions=action_seq,
    target_metric="reward_pred",  # Compare outcome in reward space
)

for idx, row in compared.items():
    print(f"Intervention {idx}: {row['intervention_description']}")
    print(f"  Baseline outcome: {row['baseline_outcome']:.6f}")
    print(f"  Counterfactual outcome: {row['counterfactual_outcome']:.6f}")
    print(f"  Outcome delta: {row['outcome_delta']:.6f}")
    print(f"  L2 divergence: {row['l2_distance']:.6f}")
    print(f"  Trajectory divergence: {row['trajectory_distance']:.6f}")
```

**Output:**
```
Intervention 0: Ablate dim 0
  Baseline outcome: 0.523456
  Counterfactual outcome: 0.412345
  Outcome delta: -0.111111
  L2 divergence: 0.234567
  Trajectory divergence: 0.345678

Intervention 1: Ablate dim 1
  Baseline outcome: 0.523456
  Counterfactual outcome: 0.487654
  Outcome delta: -0.035802
  L2 divergence: 0.123456
  Trajectory divergence: 0.234567

Intervention 2: Zero all actions at t=5
  Baseline outcome: 0.523456
  Counterfactual outcome: 0.198765
  Outcome delta: -0.324691
  L2 divergence: 0.876543
  Trajectory divergence: 0.987654
```

---

## Running the Example

```bash
python examples/10_causal_engine.py
```

Expected output:
```
============================================================
World Model Lens — CounterfactualEngine walkthrough
============================================================

Baseline trajectory: 15 states
  State dim: torch.Size([128])

Counterfactual (dimension ablation on z_posterior): done.

rollout_comparison (baseline vs counterfactual):
  reward_divergence: 0.234567
  state_divergence: 0.123456
  action_divergence: 0.345678
  value_divergence: 0.456789

trace_divergence (cumulative MSE-style drift) at sample timesteps:
  t=0: cumulative = 0.000000
  t=7: cumulative = 0.234567
  t=14: cumulative = 0.345678

BranchTree: 3 branches from baseline
  [0] fork=5 divergence=0.345678 — Ablate dim 0
  [1] fork=5 divergence=0.234567 — Ablate dim 1
  [2] fork=5 divergence=0.876543 — Zero action at t=5

compare_interventions (reward_pred outcome + divergence metrics):
  intervention 0:
    intervention_description: Ablate dim 0
    target_timestep: 5
    baseline_outcome: 0.523456
    counterfactual_outcome: 0.412345
    outcome_delta: -0.111111
    l2_distance: 0.234567
    trajectory_distance: 0.345678
  ...
```

---

## Intervention Types

### Type 1: Dimension Ablation

Zero specific latent dimensions:

```python
Intervention(
    target_timestep=5,
    target_type="dimension",
    target_indices=[0, 1, 2],  # Ablate dims 0, 1, 2
)
```

Use case: "Does this latent dimension matter?"

### Type 2: Action Intervention

Zero or modify actions:

```python
Intervention(
    target_timestep=7,
    target_type="action",
    target_indices=None,  # All dimensions
)
```

Use case: "What if we didn't move at this timestep?"

### Type 3: Component Patching

Replace component from clean run:

```python
Intervention(
    target_timestep=5,
    target_type="component",
    target_component="z_posterior",
    target_indices=None,
)
```

Use case: "Does patching clean z_posterior help?" (Exactly like Example 03)

---

## Metrics Explained

### Single Intervention Metrics

| Metric | Meaning | Range |
|--------|---------|-------|
| `reward_divergence` | Difference in reward prediction | 0–1 |
| `state_divergence` | Difference in latent state | 0–∞ |
| `action_divergence` | Difference in action distribution | 0–1 |
| `value_divergence` | Difference in value estimate | 0–∞ |

### Intervention Comparison Metrics

| Metric | Meaning |
|--------|---------|
| `baseline_outcome` | Target metric with no intervention |
| `counterfactual_outcome` | Target metric after intervention |
| `outcome_delta` | Difference (counterfactual - baseline) |
| `l2_distance` | L2 norm of state divergence |
| `trajectory_distance` | Sum of state divergences over time |

---

## Workflow: Detecting Important Components

1. **Start with a surprise peak** (from Example 05)
   - Find timestep where model was uncertain

2. **Create interventions** for each latent dimension
   ```python
   interventions = [
       Intervention(target_timestep=t, target_type="dimension", target_indices=[i])
       for i in range(latent_dim)
   ]
   ```

3. **Build branch tree**
   - See which dimensions cause largest divergence

4. **Compare interventions**
   - Measure outcome impact for each

5. **Interpret results**
   - High outcome_delta = important dimension
   - High trajectory_distance = affects downstream predictions

---

## Advanced Usage

### Custom Intervention Metrics

```python
def my_metric(baseline_trajectory, cf_trajectory):
    """Custom metric comparing trajectories."""
    baseline_states = [s.state for s in baseline_trajectory.states]
    cf_states = [s.state for s in cf_trajectory.states]
    
    # Your custom comparison
    distances = [
        (b - c).norm().item()
        for b, c in zip(baseline_states, cf_states)
    ]
    return sum(distances) / len(distances)

# Use with engine
...
```

### Chained Interventions

```python
# First intervention
cf1 = engine.intervene(obs_seq, intervention1, action_seq)

# Second intervention on the counterfactual
cf2 = engine.intervene(cf1.observations, intervention2, cf1.actions)

# Compare both
metrics = rollout_comparison(baseline_traj, cf2)
```

---

## When to Use Each Tool

| Tool | Question | Example |
|------|----------|---------|
| **Example 03: Patching** | Is this component necessary? | Patch z_posterior, measure recovery |
| **Example 04: Branching** | What are plausible futures? | Branch from t=5, compare diversity |
| **Example 10: Counterfactual** | What's the causal effect? | Ablate each dimension, measure outcome |

**Counterfactual Engine** is the most comprehensive—it subsumes patching and branching with additional comparison metrics.

---

## Summary

The CounterfactualEngine enables **systematic causal analysis**:

✅ Single interventions with detailed metrics
✅ Multiple counterfactuals in one call (branch trees)
✅ Consistent comparison across interventions
✅ Divergence tracing over time
✅ Custom metrics and complex workflows

Use this to understand the **causal structure** of your world model's decision-making.
