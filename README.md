# Boids Simulation with Q-Learning

This project implements a Boids flocking simulation enhanced with Q-learning for adaptive behavior. The simulation demonstrates how simple agents (boids) can exhibit complex flocking behavior through the interaction of three basic rules, while learning to optimize their behavior through reinforcement learning.

## Overview

The simulation combines two main components:
1. **Boids Flocking Model**: Implements the classic flocking behavior using three fundamental rules
2. **Q-Learning**: Enables boids to learn and adapt their behavior based on rewards and penalties

## Boids Rules

The simulation implements three classic boids rules:

1. **Separation**: Steer to avoid crowding local flockmates
   - Formula: F_sep = sum((p_i - p_j) / |p_i - p_j|) for all j in neighborhood
   - Where p_i is the position of the current boid and p_j are positions of neighbors

2. **Alignment**: Steer towards the average heading of local flockmates
   - Formula: F_ali = (sum(v_j) / N) - v_i
   - Where v_i is velocity of current boid, v_j are velocities of neighbors, N is number of neighbors

3. **Cohesion**: Steer to move toward the average position of local flockmates
   - Formula: F_coh = (sum(p_j) / N) - p_i
   - Where p_i is position of current boid, p_j are positions of neighbors, N is number of neighbors

## Q-Learning Implementation

Each boid implements Q-learning to optimize its behavior:

- **State Space**: Position, velocity, and acceleration of the boid
- **Actions**: Choose between separation, alignment, and cohesion
- **Reward Function**: Considers:
  - Local density (optimal distance from neighbors)
  - Velocity alignment with neighbors
  - Distance to flock center
  - Speed regulation
  - Action-specific rewards

### Learning Parameters
- Learning rate (α): 0.5
- Discount factor (γ): 0.95
- Epsilon (ε): Starts at 1.0, decays to 0.01
- Epsilon decay rate: 0.995

## Requirements

- Python 3.x
- Required packages:
  - mesa
  - numpy
  - matplotlib

## Usage

The simulation can be run with different parameters:
- Number of boids
- Simulation steps
- World dimensions
- Learning parameters

## Files

- `Boids-1.0.py`: Initial implementation of the Boids simulation with Q-learning
- `Boids-2.0.py`: Enhanced version with improved reward function and documentation

## Features

- Real-time visualization of flocking behavior
- Adaptive learning through Q-learning
- Configurable parameters for simulation and learning
- Metrics collection for analysis
- Visualization of learning progress

## Learning Process

The boids learn through:
1. State observation
2. Action selection (ε-greedy policy)
3. Reward calculation
4. Q-value updates
5. Weight adjustment for flocking rules

## Reward System

The reward function considers multiple factors:
- Positive rewards for:
  - Maintaining optimal distance from neighbors
  - Aligning velocity with the flock
  - Staying close to the flock center
  - Maintaining moderate speed
- Negative rewards for:
  - Collisions
  - Isolation from the flock
  - Extreme speeds
  - Poor alignment

## Future Improvements

Potential enhancements:
- More sophisticated state representation
- Deep Q-learning implementation
- Additional flocking rules
- Obstacle avoidance
- Predator-prey dynamics
- Multi-objective optimization 