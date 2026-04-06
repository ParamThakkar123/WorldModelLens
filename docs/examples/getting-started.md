# Getting Started with Examples

## Example 01: Quickstart

**File:** `examples/01_quickstart.py`

**What it teaches:** The fundamental workflow for interpretability analysis.

### What It Does

1. Creates a DreamerV3 world model with a configuration
2. Wraps it in `HookedWorldModel` to enable activation caching
3. Runs a forward pass, collecting activations along the way
4. Extracts and inspects cached activations
5. Uses imagination to rollout from a mid-trajectory state

### Key Concepts

- **HookedWorldModel**: Central wrapper that intercepts activations
- **Cache**: Dictionary of all intermediate activations (h_t, z_posterior, z_prior, etc.)
- **Trajectory**: The sequence of model states produced by a forward pass
- **Imagination**: Continuing from a latent state without real observations

### Code Walkthrough

```python
# Step 1: Setup
cfg = WorldModelConfig(d_h=256, n_cat=32, ...)
adapter = DreamerV3Adapter(cfg)
wm = HookedWorldModel(adapter=adapter, config=cfg)

# Step 2: Data
obs_seq = torch.randn(10, 3, 64, 64)  # 10 frames, 3 channels, 64x64
action_seq = torch.randn(10, 4)       # 10 actions, 4-dimensional

# Step 3: Forward pass with caching
traj, cache = wm.run_with_cache(obs_seq, action_seq)

# Step 4: Inspect results
print(traj.length)                 # How many states?
print(cache.component_names)       # What was cached?
h_t = cache["h", 0]               # Hidden state at t=0
z_posterior = cache["z_posterior"] # All posterior samples

# Step 5: Imagination
imagined = wm.imagine(start_state=traj.states[5], horizon=20)
```

### What the Cache Contains

For a DreamerV3 world model, the cache typically includes:

| Component | Shape | Meaning |
|-----------|-------|---------|
| `h` | (T, d_h) | Hidden state from encoder/RNN |
| `z_posterior` | (T, n_cat, n_cls) | Posterior latent distribution |
| `z_prior` | (T, n_cat, n_cls) | Prior latent (from dynamics) |
| `reward_pred` | (T, 1) | Predicted rewards |
| `value_pred` | (T, 1) | Estimated values |

Access by component name or with index: `cache["component_name", t]` or `cache["component_name"]`

### Running It

```bash
python examples/01_quickstart.py
```

Expected output:
```
============================================================
World Model Lens - Quickstart Example
============================================================

[1] Config created: d_h=256, n_cat=32, n_cls=32
[2] DreamerV3Adapter initialized
[3] HookedWorldModel wrapper created
[4] Created fake data: obs=torch.Size([10, 3, 64, 64]), actions=torch.Size([10, 4])
[5] Forward pass complete!
    Trajectory length: 10
    Cache keys: ['h', 'z_posterior', 'z_prior', 'reward_pred', 'value_pred']
[6] Sample activations:
    h_t shape: torch.Size([256])
    z_posterior shape: torch.Size([32, 32])
[7] Imagination complete: 20 steps

============================================================
Quickstart complete!
============================================================
```

### Key Takeaways

- **HookedWorldModel** intercepts activations automatically
- **cache** is a dictionary-like object with timestep indexing
- **traj.states** contains StateDict objects with latent information
- **wm.imagine()** continues from any state without observations

### Next Steps

Once comfortable with this example:
- Modify the config (try different d_h, n_cat values)
- Change the trajectory length (T)
- Extract different components from the cache
- Try longer imagination horizons

Then proceed to Example 02 to learn how to **use** these cached activations.
