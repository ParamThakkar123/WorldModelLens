"""Latent Trajectory Visualization.

PCA/t-SNE projections of latent states over time.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
import torch
import numpy as np


@dataclass
class LatentProjection:
    """2D projection of latent trajectory."""

    x: np.ndarray
    y: np.ndarray
    labels: np.ndarray  # Timestep or other label
    colors: Optional[np.ndarray] = None


class LatentTrajectoryPlotter:
    """Visualize latent trajectories in 2D.

    Example:
        plotter = LatentTrajectoryPlotter(world_model)

        # Generate trajectory
        obs = torch.randn(20, 3, 64, 64)
        traj, cache = world_model.run_with_cache(obs)

        # PCA projection
        pca = plotter.project_pca(traj, n_components=2)

        # t-SNE projection
        tsne = plotter.project_tsne(traj, perplexity=5)

        # Plot with matplotlib
        import matplotlib.pyplot as plt
        plt.scatter(pca.x, pca.y, c=pca.labels)
        plt.colorbar()
        plt.show()
    """

    def __init__(self, world_model: Any):
        """Initialize plotter.

        Args:
            world_model: HookedWorldModel instance
        """
        self.wm = world_model
        self._pca_cache = {}
        self._tsne_cache = {}

    def extract_latents(
        self,
        trajectory: Any,
        component: str = "z_posterior",
    ) -> torch.Tensor:
        """Extract latent states from trajectory.

        Args:
            trajectory: WorldTrajectory
            component: Which component to extract

        Returns:
            Tensor of shape [T, d_z]
        """
        latents = []

        for state in trajectory.states:
            if component == "z_posterior" or component == "z":
                if state.obs_encoding is not None:
                    latents.append(state.obs_encoding)
                else:
                    latents.append(state.state)
            elif component == "h" or component == "hidden":
                latents.append(state.state)
            else:
                latents.append(state.state)

        return torch.stack(latents)

    def project_pca(
        self,
        trajectory: Any,
        n_components: int = 2,
        component: str = "z_posterior",
    ) -> LatentProjection:
        """Project latent trajectory using PCA.

        Args:
            trajectory: WorldTrajectory
            n_components: Number of PCA components
            component: Which latent component to use

        Returns:
            LatentProjection with 2D coordinates
        """
        latents = self.extract_latents(trajectory, component)

        # Flatten
        if latents.dim() > 2:
            latents = latents.flatten(1)

        # Center
        mean = latents.mean(dim=0, keepdim=True)
        latents = latents - mean

        # SVD
        U, S, Vt = torch.pca_lowrank(latents, q=n_components)

        projected = torch.matmul(latents, Vt[:, :n_components])

        projected = projected.detach()

        # Labels: timesteps
        labels = np.arange(len(trajectory.states))

        return LatentProjection(
            x=projected[:, 0].numpy(),
            y=projected[:, 1].numpy(),
            labels=labels,
        )

    def project_tsne(
        self,
        trajectory: Any,
        perplexity: float = 5.0,
        n_iter: int = 1000,
        component: str = "z_posterior",
    ) -> LatentProjection:
        """Project latent trajectory using t-SNE.

        Args:
            trajectory: WorldTrajectory
            perplexity: t-SNE perplexity
            n_iter: Number of iterations
            component: Which latent component to use

        Returns:
            LatentProjection with 2D coordinates
        """
        latents = self.extract_latents(trajectory, component)

        if latents.dim() > 2:
            latents = latents.flatten(1)

        latents_np = latents.numpy()

        from sklearn.manifold import TSN

        tsne = TSN(
            n_components=2,
            perplexity=min(perplexity, len(latents_np) - 1),
            n_iter=n_iter,
            random_state=42,
        )

        projected = tsne.fit_transform(latents_np)

        labels = np.arange(len(trajectory.states))

        return LatentProjection(
            x=projected[:, 0],
            y=projected[:, 1],
            labels=labels,
        )

    def project_umap(
        self,
        trajectory: Any,
        n_neighbors: int = 5,
        min_dist: float = 0.1,
        component: str = "z_posterior",
    ) -> LatentProjection:
        """Project latent trajectory using UMAP.

        Args:
            trajectory: WorldTrajectory
            n_neighbors: UMAP n_neighbors
            min_dist: UMAP min_dist
            component: Which latent component to use

        Returns:
            LatentProjection with 2D coordinates
        """
        try:
            import umap
        except ImportError:
            raise ImportError("umap-learn not installed. Install with: pip install umap-learn")

        latents = self.extract_latents(trajectory, component)

        if latents.dim() > 2:
            latents = latents.flatten(1)

        latents_np = latents.numpy()

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42,
        )

        projected = reducer.fit_transform(latents_np)

        labels = np.arange(len(trajectory.states))

        return LatentProjection(
            x=projected[:, 0],
            y=projected[:, 1],
            labels=labels,
        )

    def color_by_reward(
        self,
        trajectory: Any,
    ) -> np.ndarray:
        """Get colors based on reward values.

        Args:
            trajectory: WorldTrajectory

        Returns:
            Array of reward values
        """
        rewards = []

        for state in trajectory.states:
            reward = state.predictions.get("reward")
            if reward is not None:
                if isinstance(reward, torch.Tensor):
                    rewards.append(reward.item())
                else:
                    rewards.append(reward)
            else:
                rewards.append(0.0)

        return np.array(rewards)

    def color_by_surprise(
        self,
        trajectory: Any,
        cache: Any,
    ) -> np.ndarray:
        """Get colors based on surprise (KL divergence).

        Args:
            trajectory: WorldTrajectory
            cache: ActivationCache

        Returns:
            Array of surprise values
        """
        surprises = []

        for t in range(len(trajectory.states)):
            kl = cache.get("kl", t)
            if kl is not None:
                if isinstance(kl, torch.Tensor):
                    surprises.append(kl.item())
                else:
                    surprises.append(kl)
            else:
                surprises.append(0.0)

        return np.array(surprises)

    def plot_trajectory_gallery(
        self,
        trajectories: List[Any],
        labels: Optional[List[str]] = None,
        method: str = "pca",
    ) -> List[LatentProjection]:
        """Plot multiple trajectories in same space.

        Args:
            trajectories: List of WorldTrajectories
            labels: Labels for each trajectory
            method: Projection method ("pca", "tsne", "umap")

        Returns:
            List of LatentProjections
        """
        if labels is None:
            labels = [f"Traj {i}" for i in range(len(trajectories))]

        projections = []

        for traj, label in zip(trajectories, labels):
            if method == "pca":
                proj = self.project_pca(traj)
            elif method == "tsne":
                proj = self.project_tsne(traj)
            elif method == "umap":
                proj = self.project_umap(traj)
            else:
                raise ValueError(f"Unknown method: {method}")

            projections.append(proj)

        return projections

    def compute_velocity(
        self,
        trajectory: Any,
        component: str = "z_posterior",
    ) -> np.ndarray:
        """Compute velocity (change) between timesteps.

        Args:
            trajectory: WorldTrajectory
            component: Which component

        Returns:
            Array of velocities
        """
        latents = self.extract_latents(trajectory, component)

        if latents.dim() > 2:
            latents = latents.flatten(1)

        velocities = torch.diff(latents, dim=0).norm(dim=1)

        return velocities.numpy()

    def compute_acceleration(
        self,
        trajectory: Any,
        component: str = "z_posterior",
    ) -> np.ndarray:
        """Compute acceleration (change in velocity).

        Args:
            trajectory: WorldTrajectory
            component: Which component

        Returns:
            Array of accelerations
        """
        latents = self.extract_latents(trajectory, component)

        if latents.dim() > 2:
            latents = latents.flatten(1)

        velocities = torch.diff(latents, dim=0)
        accelerations = torch.diff(velocities, dim=0).norm(dim=1)

        return accelerations.numpy()
