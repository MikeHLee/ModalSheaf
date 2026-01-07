#!/usr/bin/env python3
"""
Example 07: Reward Sheaves and Hodge Decomposition
==================================================

This example demonstrates how to use sheaf theory to analyze reinforcement learning 
reward structures, specifically focusing on detecting cyclic preferences (Condorcet cycles).

We simulate a "Condorcet Ring" environment where an agent gets positive reward
for moving clockwise, creating a "money pump" that standard value functions V(s)
cannot represent (because V(s) must be single-valued).

We show two approaches:
1. Discrete RewardSheaf: Detects positive loops in the state-transition graph (H¹).
2. Neural Hodge Decomposition: Learns a reward model r = dV + ω that separates
   the "rational" potential V from the "irrational" cyclic flow ω.
"""

import numpy as np
import time

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from modalsheaf.applications.rl import RewardSheaf, HodgeCritic


def run_discrete_analysis():
    print("\n" + "="*60)
    print("1. DISCRETE SHEAF ANALYSIS")
    print("="*60)
    
    if not HAS_NETWORKX:
        print("Skipping: networkx not installed")
        return

    # Simulate a Condorcet Cycle
    # States: A -> B -> C -> A
    # Reward: +1 for each step
    
    sheaf = RewardSheaf(directed=True)
    
    print("Simulating trajectory: A -> B -> C -> A (Reward +1 each step)...")
    
    # Cycle 1
    sheaf.add_transition("A", "B", reward=1.0)
    sheaf.add_transition("B", "C", reward=1.0)
    sheaf.add_transition("C", "A", reward=1.0)
    
    # Cycle 2 (with some noise)
    sheaf.add_transition("A", "B", reward=0.9)
    sheaf.add_transition("B", "C", reward=1.1)
    sheaf.add_transition("C", "A", reward=1.0)
    
    # Analyze
    cycles = sheaf.find_positive_cycles()
    
    print(f"\nFound {len(cycles)} positive cycles (Money Pumps):")
    for i, c in enumerate(cycles):
        print(f"Cycle {i+1}: {' -> '.join(map(str, c.cycle_nodes))}")
        print(f"  Total Reward: {c.total_reward:.2f}")
        print(f"  Avg Reward:   {c.avg_reward:.2f}")
        print(f"  Interpretation: H¹ obstruction detected!")

    h1_score = sheaf.compute_empirical_h1()
    print(f"\nEmpirical H¹ Score: {h1_score:.4f}")


def run_neural_hodge_decomposition():
    print("\n" + "="*60)
    print("2. NEURAL HODGE DECOMPOSITION")
    print("="*60)
    
    if not HAS_TORCH:
        print("Skipping: torch not installed")
        return

    print("Training Hodge Critic on Condorcet Ring...")
    print("Ground Truth: r = 1.0 (constant clockwise force)")
    
    # Setup
    # State is (sin(theta), cos(theta)) on unit circle
    critic = HodgeCritic(state_dim=2, hidden_dim=64, use_harmonic=True)
    optimizer = optim.Adam(critic.parameters(), lr=0.01)
    
    # Generate training data: purely clockwise motion
    # theta(t) = t * 0.1
    # reward = 1.0 * velocity
    batch_size = 100
    theta = torch.rand(batch_size) * 2 * np.pi
    velocity = 0.1  # Constant angular velocity
    
    # Next state
    theta_next = theta + velocity
    
    # Convert to (sin, cos)
    state = torch.stack([torch.sin(theta), torch.cos(theta)], dim=1)
    next_state = torch.stack([torch.sin(theta_next), torch.cos(theta_next)], dim=1)
    
    # Reward is proportional to velocity (r = v)
    # If we integrate around circle: total reward = 2pi
    # Average reward per step = 0.1
    rewards = torch.ones(batch_size) * velocity
    
    # Velocity vector for the critic (simple scalar here for 1D motion)
    # The critic expects a tensor matching the harmonic coefficient shape logic
    # In our implementation, if velocity is scalar, it works.
    velocities = torch.ones(batch_size, 1) * velocity
    
    # Train
    print("\nTraining...")
    for epoch in range(500):
        # Predict: r = dV + ω*v
        pred_rewards = critic.predict_reward(state, next_state, velocities).squeeze()
        
        # Loss: MSE + Orthogonality (V should sum to 0)
        V_curr, harmonic_coeff = critic(state)
        loss = critic.loss(pred_rewards, rewards, V_curr)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            omega = harmonic_coeff.mean().item()
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}, Learned ω = {omega:.4f}")

    # Results
    final_omega = critic.harmonic_coeff.item()
    print("\n" + "-"*30)
    print("RESULTS")
    print("-" * 30)
    print(f"True H¹ (Target ω): 1.0000")
    print(f"Learned H¹ (ω):     {final_omega:.4f}")
    print(f"Error:              {abs(1.0 - final_omega):.4f}")
    
    print("\nInterpretation:")
    if abs(1.0 - final_omega) < 0.1:
        print("SUCCESS: The critic successfully separated the cyclic reward (ω)")
        print("from the potential function V(s). Standard RL would fail here.")
    else:
        print("PARTIAL: The decomposition is approximate.")


def main():
    print("ModalSheaf: Reward Space Analysis Example")
    run_discrete_analysis()
    run_neural_hodge_decomposition()
    print("\nDone.")


if __name__ == "__main__":
    main()
