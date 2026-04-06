# Advanced Analysis

These examples demonstrate high-level interpretability analyses built on top of the core techniques.

## Example 05: Belief Analysis

**File:** `examples/05_belief_analysis.py`

**What it teaches:** How to analyze model uncertainty, find important concepts, and detect hallucinations.

### What It Does

1. Computes **surprise timeline** — when was the model uncertain?
2. Searches for **concept alignment** — which dimensions encode a concept?
3. Computes **saliency maps** — which latents affect reward prediction?
4. Detects **hallucinations** — where does imagination diverge from reality?

### Core Analyses

#### 1. Surprise Timeline

Measures model confidence using KL divergence between prior and posterior.

```python
analyzer = BeliefAnalyzer(wm)
surprise_result = analyzer.surprise_timeline(cache)

print(f"Mean surprise: {surprise_result.mean_surprise:.4f}")
print(f"Max surprise at t={surprise_result.max_surprise_timestep}")
print(f"Peaks: {surprise_result.peaks}")  # Timesteps where model was most uncertain
```

**Interpretation:**
- High surprise → Model encountered something unexpected
- Surprise peaks → Important decision points or state transitions

#### 2. Concept Search

Find which latent dimensions best discriminate two groups of timesteps.

```python
concept_result = analyzer.concept_search(
    concept_name="early_vs_late",
    positive_timesteps=[0, 1, 2, 3, 4],    # Early times
    negative_timesteps=[10, 11, 12, 13, 14],  # Late times
    cache=cache,
    component="z_posterior",
)

print(f"Top dims: {concept_result.top_dims[:5]}")
```

**Interpretation:**
- `top_dims` = latent dimensions that differ most between the two groups
- Use this to understand how the model structures state

#### 3. Saliency Maps

Compute gradients to find which latents causally affect a target (e.g., reward prediction).

```python
saliency_result = analyzer.latent_saliency(
    traj=traj,
    cache=cache,
    timestep=5,
    target="reward_pred",  # Compute gradients w.r.t. reward
)

print(f"h_saliency shape: {saliency_result.h_saliency.shape}")  # Gradients of h
print(f"z_saliency shape: {saliency_result.z_saliency.shape}")  # Gradients of z
```

**Interpretation:**
- High saliency values = strong causal effect on target
- Use to identify which parts of the state matter for a given output

#### 4. Hallucination Detection

Compare imagined vs. real trajectories to find where the model's imagination diverges.

```python
imagined = wm.imagine(start_state=traj.states[0], horizon=20)

hallucination_result = analyzer.detect_hallucinations(
    real_traj=traj,
    imagined_traj=imagined,
    method="latent_distance",
    threshold=0.5,
)

print(f"Severity score: {hallucination_result.severity_score:.4f}")
print(f"Hallucination timesteps: {hallucination_result.hallucination_timesteps}")
```

**Interpretation:**
- `severity_score` = overall divergence magnitude
- `hallucination_timesteps` = where imagination diverged most

### Running It

```bash
python examples/05_belief_analysis.py
```

Expected output:
```
[2] Computing surprise timeline...
    Mean surprise: 0.2847
    Max surprise at t=8
    Peak count: 3

[3] Searching for concept alignment...
    Top dims: [12, 3, 7, 19, 11]
    Method: activation_difference

[4] Computing saliency...
    h_saliency shape: torch.Size([128])
    z_saliency shape: torch.Size([16, 32])

[5] Detecting hallucinations...
    Severity score: 0.3142
    Hallucination timesteps: [4, 5, 6]
```

---

## Example 06: Disentanglement Analysis

**File:** `examples/06_disentanglement.py`

**What it teaches:** How to measure whether a latent representation separates different factors of variation.

### What It Does

1. Creates synthetic **factors** (speed, direction, reward level)
2. Computes disentanglement metrics (**MIG**, **DCI**, **SAP**)
3. Assigns each factor to latent dimensions
4. Reports overall disentanglement score

### Why Disentanglement?

A disentangled representation has:
- One dimension per concept
- Minimal redundancy
- Interpretability (you can identify what each dimension encodes)

### Disentanglement Metrics

#### MIG (Mutual Information Gap)

Measures if each factor is encoded in a unique dimension.

- `MIG=0`: No disentanglement
- `MIG=1`: Perfect disentanglement

#### DCI (Disentangled Components)

Measures how separated factors are across dimensions.

- Includes: explicitness (how predictable each dimension from one factor) and completeness (how many dimensions each factor uses)

#### SAP (Separated Attribute Predictability)

Another measure of factor-dimension alignment.

- `SAP=0`: Random alignment
- `SAP=1`: Each factor in one dimension

### Code Walkthrough

```python
# Create synthetic factors
factors = {
    "speed": torch.tensor([float(i % 10) / 10 for i in range(50)]),
    "direction": torch.tensor([float((i * 7) % 10) / 10 for i in range(50)]),
    "reward_level": torch.tensor([1.0 if i < 25 else 0.0 for i in range(50)]),
}

# Compute disentanglement
disentanglement_result = analyzer.disentanglement_score(
    cache=cache,
    factors=factors,
    metrics=["MIG", "DCI", "SAP"],
    component="z_posterior",
)

# Results
print(f"MIG: {disentanglement_result.scores['MIG']:.4f}")
print(f"DCI: {disentanglement_result.scores['DCI']:.4f}")
print(f"SAP: {disentanglement_result.scores['SAP']:.4f}")

# See which dimensions encode which factors
for factor, dims in disentanglement_result.factor_dim_assignment.items():
    print(f"{factor}: dims {dims[:5]}")
```

### Interpreting Results

| Score | Meaning |
|-------|---------|
| 0.0–0.3 | Low disentanglement; factors are mixed |
| 0.3–0.6 | Moderate disentanglement; some structure |
| 0.6–0.8 | Good disentanglement; clear separation |
| 0.8–1.0 | Excellent; each factor in isolated dimensions |

### Running It

```bash
python examples/06_disentanglement.py
```

Expected output:
```
[3] Computing disentanglement metrics...
    MIG score: 0.3214
    DCI score: 0.4567
    SAP score: 0.3891
    Total score: 0.3891

[4] Factor assignments:
    speed: dims [3, 7, 11, 12, 15]...
    direction: dims [1, 8, 14, 2, 9]...
    reward_level: dims [4, 6, 10, 0, 13]...
```

---

## Combining the Two Advanced Examples

| Example | Concept | Metric | Use Case |
|---------|---------|--------|----------|
| **05: Belief** | Model confidence | Surprise timeline | When is the model uncertain? |
| **05: Belief** | Saliency | Gradient importance | Which states matter? |
| **05: Belief** | Hallucination | Divergence detection | Where does imagination fail? |
| **06: Disent.** | Factor structure | MIG/DCI/SAP scores | Is representation organized? |
| **06: Disent.** | Alignment | Factor-to-dimension mapping | Which dimension = which factor? |

## Next Steps

After these two examples:
- Example 07-09 show these techniques apply to **any world model type**
- Example 10 shows **counterfactual engines** for systematic intervention

You now understand both **what** the model learns (disentanglement) and **when** it's uncertain (belief analysis).
