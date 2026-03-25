"""Temporal Attribution Maps.

Show which timestep influences which timestep.
Even if model doesn't have explicit attention.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np


@dataclass
class TemporalAttributionMatrix:
    """Matrix of temporal attributions."""

    matrix: np.ndarray  # [T_source, T_target]
    source_timesteps: np.ndarray
    target_timesteps: np.ndarray
    max_attribution: float
    total_attribution: float


class TemporalAttributionMap:
    """Visualize temporal causal influence.

    Shows which timesteps influence which downstream timesteps.
    Similar to attention maps but computed via interventions.

    Example:
        mapper = TemporalAttributionMap(world_model)

        # Compute influence matrix
        obs = torch.randn(20, 3, 64, 64)
        matrix = mapper.compute_influence_matrix(obs)

        # Get influence from specific timestep
        t5_influence = mapper.influence_from_timestep(obs, source_t=5)

        # Causal flow
        flow = mapper.trace_causal_flow(obs, source_t=0, target_t=15)
    """

    def __init__(self, world_model: Any):
        """Initialize mapper.

        Args:
            world_model: HookedWorldModel instance
        """
        self.wm = world_model
        self._cache = {}

    def compute_influence_matrix(
        self,
        observations: torch.Tensor,
        target_metric: str = "state_norm",
    ) -> TemporalAttributionMatrix:
        """Compute full influence matrix.

        Entry (i, j) shows how much timestep i influences timestep j.

        Args:
            observations: Input observations
            target_metric: Metric to measure influence on

        Returns:
            TemporalAttributionMatrix
        """
        T = observations.shape[0]

        # Get baseline
        baseline_traj, _ = self.wm.run_with_cache(observations)

        influence = np.zeros((T, T))

        # For each source timestep
        for source_t in range(T):
            # Ablate at source
            def ablate_hook(tensor, ctx):
                if ctx.timestep == source_t:
                    return tensor * 0
                return tensor

            intervened_traj, _ = self.wm.run_with_advanced_hooks(
                observations,
                hook_specs={f"t={source_t}.z": ablate_hook},
            )

            # Measure effect at each target
            for target_t in range(T):
                baseline_val = self._extract_metric(baseline_traj, target_t, target_metric)
                intervened_val = self._extract_metric(intervened_traj, target_t, target_metric)

                influence[source_t, target_t] = abs(baseline_val - intervened_val)

        source_ts = np.arange(T)
        target_ts = np.arange(T)

        return TemporalAttributionMatrix(
            matrix=influence,
            source_timesteps=source_ts,
            target_timesteps=target_ts,
            max_attribution=influence.max(),
            total_attribution=influence.sum(),
        )

    def influence_from_timestep(
        self,
        observations: torch.Tensor,
        source_t: int,
    ) -> np.ndarray:
        """Get influence from specific source timestep to all targets.

        Args:
            observations: Input observations
            source_t: Source timestep

        Returns:
            Array of influences to each target timestep
        """
        T = observations.shape[0]

        baseline_traj, _ = self.wm.run_with_cache(observations)

        def ablate_hook(tensor, ctx):
            if ctx.timestep == source_t:
                return tensor * 0
            return tensor

        intervened_traj, _ = self.wm.run_with_advanced_hooks(
            observations,
            hook_specs={f"t={source_t}.z": ablate_hook},
        )

        influences = []

        for target_t in range(T):
            baseline_val = self._extract_metric(baseline_traj, target_t, "state_norm")
            intervened_val = self._extract_metric(intervened_traj, target_t, "state_norm")
            influences.append(abs(baseline_val - intervened_val))

        return np.array(influences)

    def influence_to_timestep(
        self,
        observations: torch.Tensor,
        target_t: int,
    ) -> np.ndarray:
        """Get influence to specific target timestep from all sources.

        Args:
            observations: Input observations
            target_t: Target timestep

        Returns:
            Array of influences from each source timestep
        """
        T = observations.shape[0]

        baseline_traj, _ = self.wm.run_with_cache(observations)

        target_baseline = self._extract_metric(baseline_traj, target_t, "state_norm")

        influences = []

        for source_t in range(T):

            def ablate_hook(tensor, ctx):
                if ctx.timestep == source_t:
                    return tensor * 0
                return tensor

            intervened_traj, _ = self.wm.run_with_advanced_hooks(
                observations,
                hook_specs={f"t={source_t}.z": ablate_hook},
            )

            intervened_val = self._extract_metric(intervened_traj, target_t, "state_norm")
            influences.append(abs(target_baseline - intervened_val))

        return np.array(influences)

    def trace_causal_flow(
        self,
        observations: torch.Tensor,
        source_t: int,
        target_t: int,
    ) -> Dict[int, float]:
        """Trace causal flow from source to target.

        Uses iterative ablation to find path.

        Args:
            observations: Input observations
            source_t: Source timestep
            target_t: Target timestep

        Returns:
            Dict mapping intermediate timesteps to their importance
        """
        T = observations.shape[0]

        baseline_traj, _ = self.wm.run_with_cache(observations)

        baseline_target = self._extract_metric(baseline_traj, target_t, "state_norm")

        path_importance = {}

        # Check each intermediate timestep
        for mid_t in range(source_t + 1, target_t):

            def path_hook(tensor, ctx):
                if ctx.timestep == source_t or ctx.timestep == mid_t:
                    return tensor * 0
                return tensor

            intervened_traj, _ = self.wm.run_with_advanced_hooks(
                observations,
                hook_specs={f"t={source_t}.z": path_hook},
            )

            intervened_val = self._extract_metric(intervened_traj, target_t, "state_norm")
            path_importance[mid_t] = abs(baseline_target - intervened_val)

        return path_importance

    def find_critical_timesteps(
        self,
        observations: torch.Tensor,
        target_t: int,
        top_k: int = 3,
    ) -> List[Tuple[int, float]]:
        """Find most critical timesteps for target.

        Args:
            observations: Input observations
            target_t: Target timestep
            top_k: Number of top timesteps to return

        Returns:
            List of (timestep, importance) tuples
        """
        influences = self.influence_to_timestep(observations, target_t)

        top_indices = np.argsort(influences)[-top_k:][::-1]

        return [(int(i), float(influences[i])) for i in top_indices]

    def temporal_gradient(
        self,
        observations: torch.Tensor,
    ) -> np.ndarray:
        """Compute temporal gradient (change in influence over time).

        Args:
            observations: Input observations

        Returns:
            Array of gradients
        """
        matrix = self.compute_influence_matrix(observations)

        gradient = np.gradient(matrix.matrix, axis=1)

        return gradient

    def causal_strength(
        self,
        observations: torch.Tensor,
        source_t: int,
        target_t: int,
    ) -> float:
        """Compute causal strength between source and target.

        Args:
            observations: Input observations
            source_t: Source timestep
            target_t: Target timestep

        Returns:
            Causal strength value
        """
        influences = self.influence_from_timestep(observations, source_t)

        if target_t < len(influences):
            return influences[target_t]

        return 0.0

    def _extract_metric(
        self,
        trajectory: Any,
        timestep: int,
        metric: str,
    ) -> float:
        """Extract metric from trajectory."""
        if timestep < 0:
            timestep = len(trajectory.states) + timestep

        if timestep >= len(trajectory.states):
            timestep = len(trajectory.states) - 1

        state = trajectory.states[timestep]

        if metric == "state_norm":
            return state.state.norm().item()
        elif metric == "obs_norm":
            if state.obs_encoding is not None:
                return state.obs_encoding.norm().item()
        elif metric == "reward":
            return state.predictions.get("reward", torch.tensor(0.0)).item()

        return 0.0

    def to_attention_style(
        self,
        observations: torch.Tensor,
        normalize: bool = True,
    ) -> np.ndarray:
        """Convert to attention-style matrix (normalized, softmax-like).

        Args:
            observations: Input observations
            normalize: Whether to normalize

        Returns:
            Attention-style matrix
        """
        matrix = self.compute_influence_matrix(observations)

        attn = matrix.matrix.copy()

        if normalize:
            # Row-normalize (each source sums to 1)
            row_sums = attn.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            attn = attn / row_sums

        return attn
