# Core Analysis Techniques

These three examples cover the main interpretability methods: understanding what activations encode (probing), identifying causal roles (patching), and exploring counterfactuals (branching).

## Example 02: Linear Probing

**File:** `examples/02_probing.py`

**What it teaches:** How to train classifiers on activations to understand what they encode.

### What It Does

1. Collects activations from 5 random trajectories
2. Creates synthetic labels (3 concepts: reward_region, novel_state, high_value)
3. Trains linear probes on the z_posterior activations
4. Reports accuracy for each concept

### Why Probing?

Probing answers: **"What information is encoded in this activation?"**

Example: If you train a probe on `z_posterior` to predict "reward regions" and it gets 85% accuracy, the model's latent representation contains reward-predictive information.

### Code Walkthrough

```python
# Collect activations from multiple episodes
all_activations = []
for ep in range(5):
    obs_seq = torch.randn(20, 3, 64, 64)
    action_seq = torch.randn(20, cfg.d_action)
    traj, cache = wm.run_with_cache(obs_seq, action_seq)
    
    # Extract z_posterior activations
    z_posterior = cache["z_posterior"]
    z_flat = z_posterior.flatten(1)
    all_activations.append(z_flat)

# Stack all
activations = torch.cat(all_activations, dim=0)  # (N, latent_dim)

# Define concepts as labels
concepts = {
    "reward_region": (labels == 0).astype(np.float32),
    "novel_state": (labels == 1).astype(np.float32),
    "high_value": (labels == 2).astype(np.float32),
}

# Train probes
prober = LatentProber(seed=42)
sweep_result = prober.sweep(
    cache={"z_posterior": activations},
    activation_names=["z_posterior"],
    labels_dict=concepts,
    probe_type="linear",
)

# Results
for concept, result in sweep_result.results.items():
    print(f"{concept}: accuracy={result.accuracy:.3f}")
```

### Interpreting Results

| Accuracy | Interpretation |
|----------|-----------------|
| ~50% (random) | Activation doesn't encode this concept |
| 60-75% | Weak encoding; information is present but diffuse |
| 75-90% | Strong encoding; model explicitly uses this information |
| >90% | Very strong; critical feature for the model |

### Running It

```bash
python examples/02_probing.py
```

Expected output:
```
[2] Training linear probes...

[3] Probe Results:
    reward_region: accuracy=0.642
    novel_state: accuracy=0.578
    high_value: accuracy=0.614
```

---

## Example 03: Activation Patching

**File:** `examples/03_patching.py`

**What it teaches:** How to perform causal intervention by patching activations between clean and corrupted runs.

### What It Does

1. Creates a **clean run** with normal observations
2. Creates a **corrupted run** (observations get noise at t=5)
3. Patches activations from the clean run into the corrupted run
4. Measures "recovery rate" — how much does patching help?

### Why Patching?

Patching answers: **"Does this component causally affect the output?"**

If patching component X at timestep t recovers 70% of lost performance, that component was important at that time.

### Code Walkthrough

```python
# Clean run (normal observations)
clean_traj, clean_cache = wm.run_with_cache(obs_seq, action_seq)

# Corrupted run (observations corrupted from t=5 onwards)
obs_corrupted = obs_seq.clone()
obs_corrupted[5:] = torch.randn_like(obs_corrupted[5:])
corrupted_traj, corrupted_cache = wm.run_with_cache(obs_corrupted, action_seq)

# Setup patcher
patcher = TemporalPatcher(wm)

# Full sweep: try patching each component at each timestep
sweep_result = patcher.full_sweep(
    clean_cache=clean_cache,
    corrupted_cache=corrupted_cache,
    components=["h", "z_posterior", "z_prior"],
    metric_fn=lambda pred: pred.mean().item(),
    t_range=[5, 6, 7, 8, 9],
)

# Find most important patches
top_patches = sweep_result.top_k_patches(k=5)
for patch in top_patches:
    print(f"{patch.component}@t={patch.timestep}: recovery={patch.recovery_rate:.3f}")
```

### Understanding Recovery Rate

**Recovery Rate** = (corrupted_metric - patched_metric) / (clean_metric - corrupted_metric)

- `recovery=0`: Patching does nothing
- `recovery=0.5`: Patching recovers 50% of lost performance
- `recovery=1.0`: Patching fully restores performance (this component is critical)

### Running It

```bash
python examples/03_patching.py
```

Expected output:
```
[3] Top patches by recovery rate:
    z_posterior@t=5: recovery=0.856
    h@t=6: recovery=0.723
    z_prior@t=7: recovery=0.591
    ...
```

---

## Example 04: Imagination Branching

**File:** `examples/04_branching.py`

**What it teaches:** How to fork from a real trajectory and explore counterfactual futures.

### What It Does

1. Records a real trajectory
2. Finds a high-surprise timestep (where the model was uncertain)
3. Branches from that point with 5 different action sequences
4. Compares how the branches diverge

### Why Branching?

Branching answers: **"What are plausible futures from here?"** and **"How certain is the model?"**

If all branches stay close together, the model is confident. If they diverge quickly, the model sees uncertainty.

### Code Walkthrough

```python
# Real trajectory
real_traj, cache = wm.run_with_cache(obs_seq, action_seq)

# Find high-surprise point (using KL divergence or your metric)
kl_vals = np.random.rand(real_traj.length)  # In real cases, compute KL
fork_at = int(np.argmax(kl_vals[5:])) + 5
start_state = real_traj.states[fork_at]

# Branch multiple times
branches = []
for _ in range(5):
    actions = torch.randn(20, cfg.d_action)
    imagined = wm.imagine(start_state=start_state, actions=actions)
    branches.append(imagined)

# Measure divergence
ref_states = torch.stack([s.state for s in branches[0].states])
for i, branch in enumerate(branches[1:], 1):
    branch_states = torch.stack([s.state for s in branch.states])
    divergence = (ref_states - branch_states).norm(dim=-1)
    print(f"Branch 0 vs {i}: mean L2={divergence.mean():.4f}")
```

### Divergence Metrics

| Metric | Interpretation |
|--------|-----------------|
| Small mean L2 | Branches stay similar (high confidence) |
| Large mean L2 | Branches diverge quickly (high uncertainty) |
| Increasing L2 over time | Compound uncertainty (typical) |
| Stable L2 over time | System converges (attractor behavior) |

### Running It

```bash
python examples/04_branching.py
```

Expected output:
```
[5] Computing divergence between branches
    Branch 0 vs 1: mean L2=0.0342, max L2=0.1289
    Branch 0 vs 2: mean L2=0.0298, max L2=0.1156
    Branch 0 vs 3: mean L2=0.0401, max L2=0.1523
    Branch 0 vs 4: mean L2=0.0387, max L2=0.1478
```

---

## Comparing the Three Techniques

| Example | Question | Method | Output |
|---------|----------|--------|--------|
| **02: Probing** | What do activations encode? | Train classifier | Accuracy scores |
| **03: Patching** | What causally affects output? | Replace → measure recovery | Recovery rates |
| **04: Branching** | What are plausible futures? | Fork & compare | Divergence curves |

All three are complementary:
- **Probing** finds correlates
- **Patching** finds causal components
- **Branching** explores counterfactuals

## Next Steps

After these three examples, you'll be ready for advanced analysis in Example 05 (surprise & concepts) and Example 06 (disentanglement).
