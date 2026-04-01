# TDA Geometric Fidelity - RSSM Latent Space Experiments

Preliminary experiments for the topological diagnostics module in WML.
The core question: does the model latent space actually preserve the geometry
of the environment that it's supposed to be modelling?

Pendulum-v1 is the testbed because its true state space is S¹ × ℝ (a cylinder),
which means we have ground truth topology to compare against. That's what makes
it useful here. We can measure fidelity, not just assume it.

---

## What's in here

```
collect.py      - replay buffer + trajectory collection
tda_utils.py    - the actual TDA pipeline (ripser to persistence diagrams to W1)
experiment.py   - full sweep: latent dims × training epochs × policies
```

---

## Setup

Tested on Python 3.10. GPU optional but speeds things up a bit.

```bash
pip install -r requirements.txt
```

If ripser gives you trouble on install (it sometimes does), try:

```bash
pip install ripser --no-binary ripser
```

---

## Running

```bash
python experiment.py
```

Outputs four plots + a summary table to stdout. Checkpoints TDA metrics every
10 epochs so you can see how topology evolves during training, not just at the end.

The sweep covers latent dims [8, 16, 32] × [random policy, PD controller].
Takes ~20-30 mins on CPU, much faster with a GPU.

---

## Key results so far

- W₁ distance between true (θ, θ̇) state space and learned latent trajectories
  decreases with training but doesn't correlate cleanly with reconstruction loss.
  That's the interesting part - loss says one thing, topology says another.

- Larger latent dim doesn't straightforwardly → better geometric fidelity.
  ld=16 sometimes outperforms ld=32 on W₁ depending on policy.

- The PD controller produces richer topological structure in the latent space
  than random policy, which makes sense - it actually visits the cylinder properly.

For Rishi: H0 total persistence is logged alongside W₁ now for your ease, curious what you find, all yours.
The hypothesis (low H0 TP = collapse, high = convergence) seems plausible to me
but I haven't had time to stress test it yet. Run `experiment.py` and check the
summary table, it should be in the last two columns.

---

## Compute setup

Mix of local and Colab depending on sweep size. The config at the top
of `experiment.py` is set conservatively so it runs locally in reasonable time.
Bump `TRAIN_EPOCHS` and `N_TRUE_SAMPLES` if you have more headroom.

---

## Next steps

- H0 total persistence as a collapse/convergence signal (Rishi's idea, worth validating)
- Extend to JEPA - the CartPole invariance loss vs W₁ stability result suggests
  the diagnostic value is even clearer there
- Once the interface stabilises this moves into `analysis/topology.py` in the main WML module
