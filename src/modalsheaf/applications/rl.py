"""
Sheaf-Theoretic Reward Spaces for Reinforcement Learning.

This module provides tools for analyzing reward functions and policies using sheaf theory,
specifically focusing on:
1. Detecting cyclic preferences (H¹ cohomology)
2. Hodge decomposition of rewards into potential (global value) and flux (cyclic) components
3. Safety analysis using topological obstructions

Mathematical Framework:
    
    We model the reward function as a 1-form on the state-action manifold.
    - Exact form (dV): Gradient of a scalar value function V(s) (traditional RL)
    - Harmonic form (ω): Cyclic preferences / non-transitive rewards (H¹ obstruction)
    
    The Hodge decomposition theorem states:
        r = dV + δA + ω
    
    In RL terms:
    - dV: "Rational" potential-based reward
    - ω: "Irrational" cyclic reward (The Condorcet component)

Example Usage:
    
    >>> from modalsheaf.applications.rl import HodgeCritic, RewardSheaf
    >>> 
    >>> # 1. Neural Hodge Decomposition
    >>> critic = HodgeCritic(state_dim=2, hidden_dim=64)
    >>> dV, omega = critic(state, next_state, velocity)
    >>> 
    >>> # 2. Discrete Analysis on Trajectories
    >>> sheaf = RewardSheaf(trajectories)
    >>> h1_cycles = sheaf.compute_persistent_cycles()
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


# ==================== Neural Hodge Decomposition ====================

class HodgeCritic(nn.Module if HAS_TORCH else object):
    """
    Neural network critic that implements Hodge decomposition for rewards.
    
    Decomposes reward r(s, s') into:
    1. Exact part dV: Difference in scalar potential V(s') - V(s)
    2. Harmonic part ω: Constant topological obstruction (flux)
    
    This allows the critic to learn cyclic preferences that standard scalar
    value functions cannot represent.
    
    Args:
        state_dim: Dimension of state space
        hidden_dim: Dimension of hidden layers
        use_harmonic: Whether to learn the harmonic component (H¹)
    """
    def __init__(self, state_dim: int, hidden_dim: int = 64, use_harmonic: bool = True):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for HodgeCritic")
            
        super().__init__()
        self.use_harmonic = use_harmonic
        
        # Potential network: V(s) -> R
        self.potential_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Harmonic coefficient (The H¹ invariant)
        # In a generalized setting, this could be a vector for multiple cycles
        # For S¹, it's a scalar.
        self.harmonic_coeff = nn.Parameter(torch.zeros(1))

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Forward pass.
        
        Returns:
            potential: V(x)
            harmonic: ω (constant)
        """
        potential = self.potential_net(x)
        harmonic = self.harmonic_coeff.expand(x.shape[0], 1)
        return potential, harmonic
    
    def predict_reward(
        self, 
        state: "torch.Tensor", 
        next_state: "torch.Tensor", 
        velocity: Optional["torch.Tensor"] = None
    ) -> "torch.Tensor":
        """
        Predict reward using Hodge decomposition: r ≈ dV + ω(v)
        
        Args:
            state: Current state
            next_state: Next state
            velocity: Velocity vector (or displacement). If None, estimated as next - curr.
            
        Returns:
            Predicted scalar reward
        """
        V_curr, _ = self.forward(state)
        V_next, omega = self.forward(next_state)
        
        # Exact part: dV = V(s') - V(s)
        dV = V_next - V_curr
        
        # Harmonic part: ω(v)
        # Only added if velocity is provided or can be inferred
        if self.use_harmonic:
            if velocity is None:
                # Naive displacement - works for Euclidean spaces
                # For more complex manifolds, velocity needs to be passed explicitly
                velocity = next_state - state
                
            # For 1D harmonic form on circle, it acts on the angular velocity
            # We assume the harmonic coefficient couples with the "magnitude" of change
            # relative to the cycle. This is a simplification of ω(v).
            # If velocity is a vector, we'd need a 1-form vector ω_vec.
            # Here we assume the scalar omega scales the provided velocity magnitude/projection.
            if velocity.dim() > 1 and velocity.shape[1] > 1:
                # If velocity is vector, we treat omega as scaling the 'rotation'
                # This implementation assumes the caller provides a scalar 'velocity' 
                # that represents motion along the cycle (like angular velocity),
                # OR we take the norm/projection.
                # To keep it generic like the notebook:
                harmonic_part = omega * velocity
            else:
                harmonic_part = omega * velocity
                
            return dV + harmonic_part
        else:
            return dV

    def loss(
        self, 
        pred_reward: "torch.Tensor", 
        true_reward: "torch.Tensor", 
        potentials: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        Compute loss with orthogonality regularization.
        
        Enforces that V(s) has zero mean, pushing constant components into ω.
        """
        mse = nn.MSELoss()(pred_reward, true_reward)
        ortho = 0.01 * potentials.mean().pow(2)
        return mse + ortho


# ==================== Discrete Reward Sheaf ====================

@dataclass
class CycleResult:
    """Result of cycle analysis on the reward sheaf."""
    cycle_nodes: List[Any]
    total_reward: float
    avg_reward: float
    is_positive_loop: bool

class RewardSheaf:
    """
    Sheaf-theoretic analysis of discrete state-action graphs.
    
    Constructs a sheaf where:
    - Base space: Graph of visited states
    - Stalks: Reward values
    - Sections: Local reward consistency
    
    Used to detect "money pumps" (positive loops) and inconsistencies.
    """
    
    def __init__(self, directed: bool = True):
        if not HAS_NETWORKX:
            raise ImportError("NetworkX is required for RewardSheaf")
        self.graph = nx.DiGraph() if directed else nx.Graph()
        
    def add_transition(self, state, next_state, reward: float, action=None):
        """Add a transition to the graph."""
        # We can use hash of state representation for node ID
        u = self._to_id(state)
        v = self._to_id(next_state)
        
        # Add edge with reward attribute
        # If edge exists, we might want to average rewards or keep list
        if self.graph.has_edge(u, v):
            old_data = self.graph[u][v]
            n = old_data.get('count', 1)
            old_r = old_data.get('reward', 0)
            new_r = (old_r * n + reward) / (n + 1)
            self.graph[u][v]['reward'] = new_r
            self.graph[u][v]['count'] = n + 1
        else:
            self.graph.add_edge(u, v, reward=reward, count=1, action=action)
            
    def _to_id(self, state):
        """Convert state to hashable ID."""
        if isinstance(state, (np.ndarray, list)):
            return tuple(np.round(state, 4))
        return state

    def find_positive_cycles(self) -> List[CycleResult]:
        """
        Detect cycles with positive net reward (money pumps).
        
        These correspond to H¹ obstructions in the reward sheaf.
        """
        cycles = list(nx.simple_cycles(self.graph))
        results = []
        
        for cycle in cycles:
            # Compute path integral (sum of rewards)
            total_r = 0
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                if self.graph.has_edge(u, v):
                    total_r += self.graph[u][v]['reward']
            
            if abs(total_r) > 1e-6:
                results.append(CycleResult(
                    cycle_nodes=cycle,
                    total_reward=total_r,
                    avg_reward=total_r / len(cycle),
                    is_positive_loop=total_r > 0
                ))
                
        return sorted(results, key=lambda x: abs(x.total_reward), reverse=True)

    def compute_empirical_h1(self) -> float:
        """
        Compute a global scalar metric for H¹ (cyclic bias).
        
        Returns sum of absolute cycle returns normalized by graph size.
        This is a heuristic for the magnitude of the cohomology.
        """
        cycles = self.find_positive_cycles()
        if not cycles:
            return 0.0
            
        total_energy = sum(abs(c.total_reward) for c in cycles)
        return total_energy / self.graph.number_of_edges()
