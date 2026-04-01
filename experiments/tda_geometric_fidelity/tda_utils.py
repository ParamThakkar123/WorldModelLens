"""
tda_utils.py - persistence diagrams, Wasserstein distance, topological fingerprints

The core TDA pipeline. Designed to be usable standalone - you can point
compute_dgm at any point cloud and get a diagram back, doesn't have to
come from an RSSM or even an RL environment.

A few non-obvious decisions documented inline.
"""

import numpy as np
from persim import wasserstein as wasserstein_dist
from ripser import ripser
from sklearn.preprocessing import StandardScaler


def normalize(X):
    """
    StandardScaler before ripser.

    Without this, β₀ saturates - every point becomes its own connected
    component because the scale differences dominate the distance matrix.
    Normalising to unit variance fixes it and makes diagrams comparable
    across different latent dimensionalities.
    """
    return StandardScaler().fit_transform(X)


def compute_dgm(X, maxdim=1):
    """
    Vietoris-Rips persistence diagram up to H1 by default.

    H1 is what we care about for cylindrical topology (one loop = S¹ factor).
    H2 available if you pass maxdim=2 but adds compute time and we haven't
    found it informative for Pendulum yet.
    """
    return ripser(normalize(X), maxdim=maxdim)["dgms"]


# ── fingerprint metrics ────────────────────────────────────────────────────────


def geometric_fidelity(dgm_true, dgm_latent):
    """
    W₁ distance between H1 diagrams of true and latent point clouds.

    Lower = the latent space is more geometrically faithful to the true
    state space. This is the main metric in the sweep.

    Handles edge cases:
    - Both empty → 0.0 (trivially faithful, but also trivially uninformative)
    - One empty → sum of half-lifetimes of the nonempty one (distance to
      the diagonal, standard TDA convention)
    """
    h1_true = dgm_true[1] if len(dgm_true) > 1 else np.empty((0, 2))
    h1_latent = dgm_latent[1] if len(dgm_latent) > 1 else np.empty((0, 2))

    # ripser uses inf to mark the essential class - drop those
    h1_true = h1_true[~np.isinf(h1_true).any(axis=1)]
    h1_latent = h1_latent[~np.isinf(h1_latent).any(axis=1)]

    if len(h1_true) == 0 and len(h1_latent) == 0:
        return 0.0

    if len(h1_true) == 0 or len(h1_latent) == 0:
        nonempty = h1_true if len(h1_latent) == 0 else h1_latent
        return float(np.sum((nonempty[:, 1] - nonempty[:, 0]) / 2))

    return float(wasserstein_dist(h1_true, h1_latent))


def loop_strength(dgm):
    """
    Max bar lifetime in H1 - how prominent is the most persistent loop.

    For Pendulum this should be non-trivial in the true state space
    (it's a cylinder) and we track whether the latent space learns that.
    Returns 0.0 if no finite H1 bars exist.
    """
    if len(dgm) < 2 or len(dgm[1]) == 0:
        return 0.0
    finite = [d for d in dgm[1] if not np.isinf(d[1])]
    return float(max(d[1] - d[0] for d in finite)) if finite else 0.0


def h0_total_persistence(dgm):
    """
    Sum of all finite H0 bar lifetimes.

    Rishi's idea - hypothesis is that this tracks collapse/convergence
    during training better than H1 metrics, because H0 reflects connected
    component structure which degrades early when representations collapse.

    High value = fragmented/spread out point cloud (possibly rich structure,
    possibly noise). Low value = tightly clustered (convergence or collapse
    depending on context). Worth validating against known collapse scenarios.

    TODO: compare against JEPA invariance loss collapse case to see if this
    actually fires where we expect it to.
    """
    if len(dgm) == 0 or len(dgm[0]) == 0:
        return 0.0
    finite = [d for d in dgm[0] if not np.isinf(d[1])]
    return float(sum(d[1] - d[0] for d in finite)) if finite else 0.0


def topological_fingerprint(dgm):
    """
    Compact summary of a persistence diagram.

    Returns a dict with the key scalar summaries - enough to compare two
    diagrams without storing the full point cloud. Useful for logging.
    """
    return {
        "h0_total_persistence": h0_total_persistence(dgm),
        "h1_loop_strength": loop_strength(dgm),
        "h1_bar_count": len([d for d in dgm[1] if not np.isinf(d[1])]) if len(dgm) > 1 else 0,
    }
