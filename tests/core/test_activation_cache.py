"""Tests for ActivationCache."""

import pytest
import torch
from world_model_lens.core.activation_cache import CacheQuery, ActivationCache


def test_single_indexing(fake_cache):
    """Test single element indexing."""
    val = fake_cache["state", 0]
    assert isinstance(val, torch.Tensor)
    assert val.shape[0] == 32


def test_slice_indexing(fake_cache):
    """Test slice indexing."""
    vals = fake_cache["state", :]
    assert vals.shape[0] == 10
    assert vals.shape[1] == 32


def test_component_names(fake_cache):
    """Test component names."""
    names = fake_cache.component_names
    assert "state" in names
    assert "posterior" in names


def test_filter(fake_cache):
    """Test filtering by component."""
    filtered = fake_cache.filter("state")
    assert "state" in filtered.component_names
    assert len(filtered.component_names) > 0


def test_to_device(fake_cache):
    """Test moving to device."""
    device = torch.device("cpu")
    moved = fake_cache.to_device(device)
    val = moved["state", 0]
    assert val.device == device


def test_detach(fake_cache):
    """Test detaching."""
    fake_cache["state", 0].requires_grad = True
    detached = fake_cache.detach()
    assert not detached["state", 0].requires_grad


def test_cachequery_stack_and_diff(fake_cache):
    """Test CacheQuery.stack and diff against a shifted cache."""
    q = CacheQuery(fake_cache)
    stacked = q.stack("state")
    assert stacked.shape[0] == 10
    # construct a second cache with +1 added to each state
    other = ActivationCache()
    for t in range(10):
        other["state", t] = fake_cache["state", t] + 1.0

    diff = q.diff(other, "state")
    # original - (original + 1) == -1
    assert diff.shape == stacked.shape
    assert torch.allclose(diff, -torch.ones_like(diff))


def test_cachequery_topk_and_correlation():
    """Test top_k_timesteps and correlation behavior with controlled data."""
    cache = ActivationCache()
    # create predictable streams
    for t in range(5):
        cache["a", t] = torch.tensor([float(t)])
        cache["b", t] = torch.tensor([float(2 * t)])

    q = CacheQuery(cache)
    top = q.top_k_timesteps("a", 2, reduce="norm")
    # largest norms are timesteps 4 and 3
    assert top[0] == 4
    assert top[1] == 3

    # correlation between a and b should be 1.0 (perfect linear)
    r = q.correlation("a", "b", reduce="mean")
    assert torch.isclose(r, torch.tensor(1.0))


def test_temporal_variability_and_most_variable_timesteps():
    """Deterministic temporal variability and most-variable timesteps."""
    cache = ActivationCache()
    # squared sequence: 0,1,4,9,16 -> diffs 1,3,5,7
    for t in range(5):
        cache["x", t] = torch.tensor([float(t * t)])

    vari = cache.temporal_variability("x")
    assert vari.shape[0] == 4
    assert torch.allclose(vari, torch.tensor([1.0, 3.0, 5.0, 7.0]))

    top2 = cache.most_variable_timesteps("x", top_k=2)
    # largest changes are 7 (timestep 4) and 5 (timestep 3)
    assert top2 == [4, 3]


def test_timesteps_exceeding_surprise():
    """timesteps_exceeding_surprise should match surprise() thresholding."""
    cache = ActivationCache()
    # prior always uniform
    prior = torch.tensor([0.5, 0.5])
    for t in range(4):
        cache["z_prior", t] = prior.clone()
        if t % 2 == 0:
            cache["z_posterior", t] = torch.tensor([0.9, 0.1])
        else:
            cache["z_posterior", t] = torch.tensor([0.5, 0.5])

    kl = cache.surprise()
    assert kl.shape[0] == 4
    exceed = cache.timesteps_exceeding_surprise(threshold=1e-6)
    # even timesteps (0 and 2) should have non-zero KL
    assert exceed == [0, 2]
