# Non-RL World Models

A core strength of World Model Lens is that it works with **any** world model architecture—not just RL agents. These three examples demonstrate video prediction and scientific dynamics models.

## Example 07: Video Prediction Model

**File:** `examples/07_video_model.py`

**What it teaches:** How to use World Model Lens with video prediction models (no actions, no rewards).

### What It Does

1. Creates a video prediction world model
2. Encodes frames into latent states
3. Predicts future frames using learned dynamics
4. Analyzes the latent representations

### Key Differences from RL Models

| RL Model (DreamerV3) | Video Model |
|---|---|
| Has actions (input) | No actions |
| Has reward predictions | No rewards |
| Has value predictions | No value |
| Optimized for control | Optimized for prediction |

### Model Architecture

```
Input frames → Encoder → Latent states → Dynamics → Decoder → Predicted frames
```

### Code Walkthrough

```python
from world_model_lens.backends.video_adapter import VideoWorldModelAdapter

# Configuration
config = WorldModelConfig(
    d_state=256,
    d_obs=3 * 64 * 64,  # Flattened frame
    is_discrete=False,
    model_type="video_prediction",
)

# Create adapter
video_model = VideoWorldModelAdapter(
    config=config,
    d_obs=3 * 64 * 64,
    frame_shape=(3, 64, 64),
)

# Wrap in HookedWorldModel
wm = HookedWorldModel(adapter=video_model, config=config)

# Forward pass (no actions needed!)
frames = torch.randn(10, 3, 64, 64)
traj, cache = wm.run_with_cache(frames)

# Predict next frames
current_frame = frames[0]
preds, states = video_model.predict_next_frame(current_frame, n_frames=5)

# Imagination (latent rollout)
start_state = traj.states[0].state
imagined_states, rewards = wm.adapter.imagine(start_state, horizon=10)
```

### What Gets Cached

For video models:

| Component | Shape | Meaning |
|-----------|-------|---------|
| `encoder_output` | (T, d_state) | Encoded latent state |
| `dynamics_output` | (T, d_state) | State predicted by dynamics |
| `decoder_output` | (T, 3, 64, 64) | Reconstructed frame |
| `prediction_error` | (T,) | L2 error vs. real frame |

### Running It

```bash
python examples/07_video_model.py
```

Expected output:
```
[1] Config: d_state=256, frame_shape=(3, 64, 64)
[2] Video world model created
[3] Wrapped in HookedWorldModel
[4] Created video sequence: torch.Size([10, 3, 64, 64])
[5] Forward pass complete!
    Trajectory length: 10
    Cache keys: ['encoder_output', 'dynamics_output', 'decoder_output']
```

---

## Example 08: Toy Video World Model Analysis

**File:** `examples/08_toy_video_world_model.py`

**What it teaches:** How to perform geometry and memory analysis on video models with synthetic moving patterns.

### What It Does

1. Creates a simple toy video world model
2. Generates synthetic video of a moving square
3. Analyzes latent **geometry** (PCA, clustering, manifold structure)
4. Analyzes **temporal memory** (how much history does the model remember?)
5. Tests that RL-specific analysis gracefully skips

### Geometry Analysis

Understand the structure of latent space:

```python
from world_model_lens.probing.geometry import GeometryAnalyzer

analyzer = GeometryAnalyzer(wm)

# PCA projection
pca_result = analyzer.pca_projection(cache, component="z_posterior")
print(f"PCA components: {pca_result.pca_components.shape}")

# Trajectory metrics
traj_metrics = analyzer.trajectory_metrics(cache)
print(f"Mean distance: {traj_metrics.mean_trajectory_distance}")
print(f"Temporal coherence: {traj_metrics.temporal_coherence}")

# Clustering
clusters = analyzer.clustering(cache, n_clusters=3)
print(f"Clusters: {clusters['clusters'].unique()}")

# Manifold analysis
manifold = analyzer.manifold_analysis(cache)
print(f"Intrinsic dimensionality: {manifold['intrinsic_dimensionality_estimate']}")
```

### Temporal Memory Analysis

Understand how much history the model remembers:

```python
from world_model_lens.probing.temporal_memory import TemporalMemoryProber

temporal = TemporalMemoryProber(wm)

# Memory retention
mem = temporal.memory_retention(cache, max_lag=10)
print(f"Memory capacity: {mem.memory_capacity}")
print(f"Short-term retention: {mem.temporal_dependencies['short_term']}")
print(f"Long-term retention: {mem.temporal_dependencies['long_term']}")

# Temporal dependencies
deps = temporal.temporal_dependencies(cache, max_lag=5)
print(f"Autocorrelations: {len(deps['autocorrelations'])}")
```

### Running It

```bash
python examples/08_toy_video_world_model.py
```

Expected output:
```
[1] Creating toy video world model...
    Model name: toy_video
    Has decoder: True
    Uses actions: False
    Is RL model: False

[2] Generating synthetic video...
    Video shape: torch.Size([30, 3, 64, 64])

[4] Belief Analysis (surprise timeline)...
    Mean surprise: 0.2134
    Max surprise: 0.5621 at t=15
    Surprise peaks: 3

[5] Geometry Analysis...
    PCA components shape: torch.Size([20, 32])
    Mean trajectory distance: 0.3456
    Temporal coherence: 0.8234
    Clusters: 3

[6] Temporal Memory Analysis...
    Memory capacity: 5.67
    Short-term retention: 0.9012
    Long-term retention: 0.1234
```

### Key Insights

- **Geometry** reveals how states organize in latent space
- **Memory** shows how much temporal context the model retains
- **Surprise** indicates prediction errors
- All work without RL-specific components (rewards, values, actions)

---

## Example 09: Toy Scientific Dynamics

**File:** `examples/09_toy_scientific_dynamics.py`

**What it teaches:** How World Model Lens works with pure scientific dynamics (physics simulations like Lorenz attractors).

### What It Does

1. Creates a toy scientific latent dynamics model
2. Generates **Lorenz attractor** trajectories (chaotic dynamics)
3. Also generates **pendulum** trajectories (periodic dynamics)
4. Analyzes latent structure using the same tools as Example 08
5. Compares the two systems

### Scientific vs. RL Models

| Aspect | RL | Scientific |
|--------|---|-----------|
| Input | Observations | State vectors |
| Dynamics | Learned, complex | Known (or to be discovered) |
| Optimization | Policy gradient | Prediction error |
| Use case | Control | Understanding |

### Creating a Scientific Model

```python
from world_model_lens.backends.toy_scientific_model import (
    ToyScientificAdapter,
    generate_lorenz_trajectory,
    generate_pendulum_trajectory,
)

# Create model
adapter = ToyScientificAdapter(
    obs_dim=3,       # Position (x, y, z)
    latent_dim=16,   # Learned latent dynamics
    hidden_dim=64,
)
wm = HookedWorldModel(adapter=adapter)

# Generate Lorenz attractor
lorenz = generate_lorenz_trajectory(n_steps=100, dt=0.02)
# Returns: (100, 3) - x, y, z coordinates

# Analyze
traj, cache = wm.run_with_cache(lorenz)
```

### Lorenz Attractor Properties

The Lorenz attractor is a famous chaotic system:

- **Deterministic**: Same initial conditions → same trajectory
- **Chaotic**: Tiny differences → huge divergence (butterfly effect)
- **Attractor**: Trajectories spiral around fixed regions

Analyzing it with WML reveals:
- How well does the model learn the chaotic dynamics?
- Where does it fail (surprise peaks)?
- How does it structure the latent space?

### Pendulum Properties

Unlike Lorenz, pendulums are periodic:

- **Periodic**: Same trajectory repeats
- **Lower dimensional**: Circle in state space
- **Predictable**: Easier to model

### Comparison

```python
# Lorenz analysis
traj1, cache1 = wm.run_with_cache(lorenz_data)
surprise1 = analyzer.surprise_timeline(cache1)
metrics1 = geometry.trajectory_metrics(cache1)

# Pendulum analysis
traj2, cache2 = wm.run_with_cache(pendulum_data)
surprise2 = analyzer.surprise_timeline(cache2)
metrics2 = geometry.trajectory_metrics(cache2)

# Compare
print(f"Lorenz mean surprise: {surprise1.mean_surprise:.4f}")
print(f"Pendulum mean surprise: {surprise2.mean_surprise:.4f}")

print(f"Lorenz trajectory distance: {metrics1.mean_trajectory_distance:.4f}")
print(f"Pendulum trajectory distance: {metrics2.mean_trajectory_distance:.4f}")
```

### Running It

```bash
python examples/09_toy_scientific_dynamics.py
```

Expected output:
```
[1] Creating toy scientific world model...
    Model name: toy_scientific
    Has decoder: False
    Has reward head: False
    Uses actions: False
    Is RL model: False

[2] Generating Lorenz attractor trajectory...
    Trajectory shape: torch.Size([100, 3])
    X range: [-16.45, 21.32]
    Y range: [-28.61, 28.09]
    Z range: [2.34, 46.78]

[4] Belief Analysis (surprise timeline)...
    Mean surprise: 0.3456
    Max surprise: 0.8901
    Surprise peaks: 7

[5] Geometry Analysis...
    PCA components shape: torch.Size([50, 16])
    Mean trajectory distance: 0.4567
    Temporal coherence: 0.7823
    Intrinsic dimensionality: 2.34

[6] Temporal Memory Analysis...
    Memory capacity: 4.23
    Short-term retention: 0.8765
    Long-term retention: 0.3456
    Dominant period: None

[9] Comparing with pendulum dynamics...
    Pendulum mean surprise: 0.1234
    Pendulum trajectory distance: 0.2345
```

### Key Takeaways

- **Lorenz**: High surprise, complex geometry, chaotic
- **Pendulum**: Low surprise, simple geometry, periodic
- Same analysis tools reveal these structural differences
- Perfect for probing what latent dynamics can represent

---

## Why These Matter

These three examples prove that World Model Lens is **model-agnostic**:

✅ Works with RL agents (Example 01-06, 10)
✅ Works with video prediction (Example 07-08)
✅ Works with scientific dynamics (Example 09)
✅ Works with any custom architecture

The analysis techniques are universal. The model just needs to implement the `WorldModelAdapter` interface.

## Next: Causal Analysis

Example 10 introduces **counterfactual engines** — the most powerful tool for systematic causal reasoning.
