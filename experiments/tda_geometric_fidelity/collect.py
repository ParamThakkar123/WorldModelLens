"""
collect.py - replay buffer + trajectory/state collection for Pendulum-v1

Nothing exotic here. The one thing worth noting: sample_true_state_space
returns (θ, θ̇) rather than the raw (cos θ, sin θ, θ̇) observation the env
gives you. That matters for TDA - (cos, sin, θ̇) lives in ℝ³ and obscures
the cylindrical topology. Extracting θ via arctan2 gives you the actual
state space geometry you want to measure against.
"""

import random
from collections import deque

import gymnasium as gym
import numpy as np

ENV_NAME = "Pendulum-v1"


class ReplayBuffer:
    def __init__(self, capacity=20_000):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, next_obs):
        self.buf.append((obs, action, next_obs))

    def sample(self, n):
        batch = random.sample(self.buf, min(n, len(self.buf)))
        obs, act, nobs = zip(*batch)
        return (
            np.array(obs, dtype=np.float32),
            np.array(act, dtype=np.float32),
            np.array(nobs, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


def fill_buffer(buf, n_steps=8_000):
    env = gym.make(ENV_NAME)
    obs, _ = env.reset()
    for _ in range(n_steps):
        action = env.action_space.sample()
        next_obs, _, term, trunc, _ = env.step(action)
        buf.push(obs, action, next_obs)
        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()
    env.close()


def sample_true_state_space(policy_fn, n_samples=300):
    """
    Collect ground-truth (θ, θ̇) pairs from actual env rollouts.

    Returns shape (n_samples, 2). This is the reference geometry -
    the thing we're asking the latent space to be faithful to.
    """
    env = gym.make(ENV_NAME)
    obs, _ = env.reset()
    states = []

    while len(states) < n_samples:
        action = policy_fn(obs)
        obs, _, term, trunc, _ = env.step(action)
        cos_th, sin_th, thdot = obs
        theta = np.arctan2(sin_th, cos_th)  # back to angle, ∈ (-π, π]
        states.append([theta, thdot])
        if term or trunc:
            obs, _ = env.reset()

    env.close()
    return np.array(states[:n_samples], dtype=np.float32)


def get_latents(enc, rssm, dec, latent_dim, hidden_dim, policy_fn, n_steps=300, device="cpu"):
    """
    Run a rollout and collect latent representations [h, z] at each step.
    These are what we run TDA on - the model's internal geometry.
    """
    import torch

    env = gym.make(ENV_NAME)
    obs, _ = env.reset()

    h = torch.zeros(1, hidden_dim).to(device)
    z = torch.zeros(1, latent_dim).to(device)
    latents = []

    for _ in range(n_steps):
        action = policy_fn(obs)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        act_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            em, elv = enc(obs_t)
            enc_z = em + torch.exp(0.5 * elv) * torch.randn_like(em)
            h, _, _ = rssm(z, act_t, h)
            z = enc_z

        latents.append(torch.cat([h, z], dim=-1).cpu().numpy().squeeze())
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()

    env.close()
    return np.array(latents)


# ── policies ──────────────────────────────────────────────────────────────────


def random_policy(obs):
    return np.array([np.random.uniform(-2, 2)])


def pd_policy(obs):
    # simple PD on angle - visits the state space more meaningfully than random
    cos_th, sin_th, thdot = obs
    theta = np.arctan2(sin_th, cos_th)
    return np.clip([-5.0 * theta - 0.5 * thdot], -2.0, 2.0)


POLICIES = {
    "random": random_policy,
    "pd_ctrl": pd_policy,
}
