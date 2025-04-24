"""
Boids Simulation with Q-Learning

This simulation implements the Boids flocking model with Q-learning for adaptive behavior.
The three classic boids rules are implemented with the following equations:

1. Separation: Steer to avoid crowding local flockmates
   F_sep = sum((p_i - p_j) / |p_i - p_j|) for all j in neighborhood
   where p_i is the position of the current boid and p_j are positions of neighbors

2. Alignment: Steer towards the average heading of local flockmates
   F_ali = (sum(v_j) / N) - v_i
   where v_i is velocity of current boid, v_j are velocities of neighbors, N is number of neighbors

3. Cohesion: Steer to move toward the average position of local flockmates
   F_coh = (sum(p_j) / N) - p_i
   where p_i is position of current boid, p_j are positions of neighbors, N is number of neighbors

The final acceleration for each boid is:
a = w_sep * F_sep + w_ali * F_ali + w_coh * F_coh
where w_sep, w_ali, w_coh are learned weights through Q-learning
"""

import mesa
import numpy as np
import random
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import matplotlib.pyplot as plt

class BoidAgent(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.position = np.array([random.uniform(0, 1), random.uniform(0, 1)])
        self.velocity = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        self.acceleration = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])  # Changed from np.zeros(2)
        self.weights = np.array([1.0, 1.0, 1.0])  # weights for separation, alignment, and cohesion
        self.states= set()

        self.epsilon = 1.0  # Initialize epsilon
        self.epsilon_decay = 0.995  # Set the decay rate for epsilon
        self.epsilon_min = 0.01  # Set the minimum value for epsilon


        #initalise q-table
        self.q_table = {}

        # Initialize learning_rate
        self.learning_rate = 0.5  # You can set this to any value between 0 and 1
        self.discount_factor = 0.95  # You can set this to any value between 0 and 1

    def choose_action(self, state):
        # Convert the state to a tuple
        state = tuple(state)

        # Implement epsilon-greedy strategy
        if random.uniform(0, 1) < self.epsilon:
            # Choose a random action
            action = random.choice(['separation', 'alignment', 'cohesion'])
        else:
            #check if the state is in the q-table
            if state not in self.q_table:
                # Add the state to the q-table
                self.q_table[state] = {'separation': 0, 'alignment': 0, 'cohesion': 0}

            # Choose a random action from the actions with the highest Q-value
            max_q_value = max(self.q_table[state].values())
            best_actions = [action for action, q_value in self.q_table[state].items() if q_value == max_q_value]
            action = random.choice(best_actions)

            # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return action

    def update_q_value(self, old_state, action, reward, new_state, alpha=0.5, gamma=0.95):
        # Check if the old state is in the Q-table
        if old_state not in self.q_table:
            # If the old state is not in the Q-table, initialize it with zero Q-values for all actions
            self.q_table[old_state] = {'separation': 0, 'alignment': 0, 'cohesion': 0}

        # Implement Q-value update
        old_q_value = self.q_table[old_state][action]
        max_new_q_value = max(self.q_table[new_state].values())
        self.q_table[old_state][action] = old_q_value + alpha * (reward + gamma * max_new_q_value - old_q_value)

    def get_state(self):
        state = (
            tuple(self.position),
            tuple(self.velocity),
            tuple(self.acceleration)
        )
        self.states.add(state)
        return state

    def reward(self, state, action):
        """Calculate reward based on how well the boid follows flocking rules.
        
        The reward function considers:
        1. Local density (not too close, not too far)
        2. Velocity alignment with neighbors
        3. Distance to flock center
        4. Speed regulation
        """
        reward = 0
        nearby_boids = self.model.space.get_neighbors(self.position, 5, False)
        
        if not nearby_boids:
            return -5  # Heavy penalty for being isolated
            
        # 1. Local density reward
        distances = [np.linalg.norm(self.position - boid.position) for boid in nearby_boids]
        avg_distance = np.mean(distances)
        if 2 < avg_distance < 4:  # Ideal distance range
            reward += 2
        elif avg_distance <= 1:  # Too close
            reward -= 3
        elif avg_distance >= 6:  # Too far
            reward -= 2
            
        # 2. Velocity alignment reward
        neighbor_velocities = [boid.velocity for boid in nearby_boids]
        avg_velocity = np.mean(neighbor_velocities, axis=0)
        velocity_alignment = np.dot(self.velocity, avg_velocity) / (np.linalg.norm(self.velocity) * np.linalg.norm(avg_velocity))
        reward += velocity_alignment  # Ranges from -1 to 1
        
        # 3. Distance to flock center reward
        flock_center = np.mean([boid.position for boid in nearby_boids], axis=0)
        distance_to_center = np.linalg.norm(self.position - flock_center)
        if distance_to_center < 3:
            reward += 1
        elif distance_to_center > 6:
            reward -= 1
            
        # 4. Speed regulation reward
        speed = np.linalg.norm(self.velocity)
        if 1 < speed < 2:  # Ideal speed range
            reward += 1
        else:
            reward -= abs(speed - 1.5)  # Penalty proportional to deviation from ideal speed
            
        # Action-specific rewards
        if action == 'separation' and avg_distance < 2:
            reward += 2  # Reward separation when too close
        elif action == 'alignment' and velocity_alignment < 0.5:
            reward += 2  # Reward alignment when misaligned
        elif action == 'cohesion' and distance_to_center > 4:
            reward += 2  # Reward cohesion when far from center
            
        return reward

    def update_weights(self, action):
        # Update the weights based on the action
        if action == 'separation':
            self.weights[0] += 1
        elif action == 'alignment':
            self.weights[1] += 1
        elif action == 'cohesion':
            self.weights[2] += 1

    def step(self):
        # Get the current state
        old_state = self.get_state()

        # Choose an action
        action = self.choose_action(old_state)

        # Get the reward based on the current state and action
        reward = self.reward(old_state, action)



        # Get the new state after taking the action
        new_state = self.get_state()

        # Update the Q-value for the current state and action
        self.update_q_value(old_state, action, reward, new_state)

        if new_state not in self.q_table:
            # If the new state is not in the Q-table, initialize it with zero Q-values for all actions
            self.q_table[new_state] = {'separation': 0, 'alignment': 0, 'cohesion': 0}

        # Calculate the maximum Q-value for the new state
        max_new_state_q_value = max(self.q_table[new_state].values())

        # Update the Q-value for the current state and action
        self.q_table[old_state][action] = (1 - self.learning_rate) * self.q_table[old_state][action] + self.learning_rate * (
                reward + self.discount_factor * max_new_state_q_value)

        # Update the weights based on the action
        self.update_weights(action)

        # Get all boids within a certain radius
        nearby_boids = self.model.space.get_neighbors(self.position, 5, False)

        # Calculate the forces
        separation_force = self.separation(nearby_boids, 1)
        alignment_force = self.alignment(nearby_boids, 5)
        cohesion_force = self.cohesion(nearby_boids, 5)

        # Update the acceleration based on the forces
        self.acceleration = self.weights[0] * separation_force + self.weights[1] * alignment_force + self.weights[
            2] * cohesion_force

        # Update the position and velocity based on the new acceleration
        self.position += self.velocity
        self.velocity += self.acceleration

    #DEFINE THE RULES OF THE SYSTEM
    def separation(self, boids, distance):
        total = 0
        steering = np.zeros(2, dtype=float)
        for boid in boids:
            if np.linalg.norm(self.position - boid.position) < distance:
                steering += (self.position - boid.position)
                total += 1
        if total > 0:
            steering /= total
        return steering

    def collision_avoidance(self, boids, min_distance):
        total = 0
        steering = np.array([0.0, 0.0])
        for boid in boids:
            distance = np.linalg.norm(self.position - boid.position)
            if distance < min_distance:
                steering += (self.position - boid.position)
                total += 1
        if total > 0:
            steering /= total
        return steering

    def cohesion(self, boids, perception_radius):
        total = 0
        center_of_mass = np.array([0.0, 0.0])
        for boid in boids:
            distance = np.linalg.norm(self.position - boid.position)
            if distance < perception_radius:
                center_of_mass += boid.position
                total += 1
        if total > 0:
            center_of_mass /= total
            return center_of_mass - self.position
        else:
            return np.array([0.0, 0.0])

    def alignment(self, boids, perception_radius):
        total = 0
        average_velocity = np.array([0.0, 0.0])
        for boid in boids:
            distance = np.linalg.norm(self.position - boid.position)
            if distance < perception_radius:
                average_velocity += boid.velocity
                total += 1
        if total > 0:
            average_velocity /= total
            return average_velocity - self.velocity
        else:
            return np.array([0.0, 0.0])

class BoidModel(mesa.Model):
    def __init__(self, width, height, num_boids):
        super().__init__()
        self.width = width
        self.height = height
        self.num_boids = num_boids
        self.space = ContinuousSpace(self.width, self.height, torus=False)
        self.schedule = RandomActivation(self)
        self.q_values_over_time = []
        
        # Metrics for flock behavior
        self.cohesion = 0
        self.alignment = 0
        self.separation = 0
        self.centering = 0
        self.avg_reward = 0
        self.flock_density = 0
        self.flock_polarization = 0  # Measure of velocity alignment
        
        # Create boids
        for i in range(self.num_boids):
            boid = BoidAgent(i, self)
            self.space.place_agent(boid, (self.random.uniform(0, self.width), self.random.uniform(0, self.height)))
            self.schedule.add(boid)
            
        # Enhanced data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Average_Reward": lambda m: m.avg_reward,
                "Flock_Density": lambda m: m.flock_density,
                "Flock_Polarization": lambda m: m.flock_polarization,
                "Separation": lambda m: m.separation,
                "Alignment": lambda m: m.alignment,
                "Cohesion": lambda m: m.cohesion,
                "Average_Q_Value": lambda m: np.mean([
                    np.mean(list(agent.q_table[state].values()))
                    for agent in m.schedule.agents
                    for state in agent.q_table
                ]) if m.schedule.agents else 0,
                "Exploration_Rate": lambda m: np.mean([
                    agent.epsilon for agent in m.schedule.agents
                ])
            },
            agent_reporters={
                "Position": lambda a: a.position.tolist(),
                "Velocity": lambda a: a.velocity.tolist(),
                "Epsilon": lambda a: a.epsilon,
                "Last_Reward": lambda a: a.reward(a.get_state(), 'separation'),  # Sample reward
                "Q_Values": lambda a: {
                    str(state): values
                    for state, values in list(a.q_table.items())[:5]  # Only store first 5 states to save memory
                }
            }
        )

    def calculate_metrics(self):
        """Calculate various metrics to measure flock behavior."""
        agents = self.schedule.agents
        positions = np.array([agent.position for agent in agents])
        velocities = np.array([agent.velocity for agent in agents])
        
        # Calculate flock density (average distance between boids)
        distances = []
        for i, pos1 in enumerate(positions):
            for pos2 in positions[i+1:]:
                distances.append(np.linalg.norm(pos1 - pos2))
        self.flock_density = np.mean(distances) if distances else 0
        
        # Calculate polarization (velocity alignment)
        avg_velocity = np.mean(velocities, axis=0)
        self.flock_polarization = np.mean([
            np.dot(v, avg_velocity) / (np.linalg.norm(v) * np.linalg.norm(avg_velocity))
            for v in velocities
        ])
        
        # Calculate average reward
        self.avg_reward = np.mean([
            agent.reward(agent.get_state(), 'separation')  # Sample reward
            for agent in agents
        ])
        
        # Update other metrics
        self.calculate_cohesion()
        self.calculate_alignment()
        self.calculate_separation()
        self.calculate_centering()

    def step(self):
        """Execute one step of the model."""
        self.schedule.step()
        self.calculate_metrics()
        self.datacollector.collect(self)

    def calculate_cohesion(self):
        """Calculate cohesion metric (average distance between boids)."""
        distances = [np.linalg.norm(agent1.position - agent2.position)
                    for agent1 in self.schedule.agents
                    for agent2 in self.schedule.agents
                    if agent1 != agent2]
        self.cohesion = np.mean(distances) if distances else 0

    def calculate_alignment(self):
        """Calculate alignment metric (variance in velocities)."""
        velocities = [agent.velocity for agent in self.schedule.agents]
        if velocities:
            velocities = np.array(velocities)
            self.alignment = np.mean([
                np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                for i, v1 in enumerate(velocities)
                for v2 in velocities[i+1:]
            ])
        else:
            self.alignment = 0

    def calculate_separation(self):
        """Calculate separation metric (minimum distance between boids)."""
        distances = [np.linalg.norm(agent1.position - agent2.position)
                    for agent1 in self.schedule.agents
                    for agent2 in self.schedule.agents
                    if agent1 != agent2]
        self.separation = np.min(distances) if distances else 0

    def calculate_centering(self):
        """Calculate centering metric (variance in positions)."""
        positions = [agent.position for agent in self.schedule.agents]
        if positions:
            positions = np.array(positions)
            center = np.mean(positions, axis=0)
            self.centering = np.mean([np.linalg.norm(pos - center) for pos in positions])
        else:
            self.centering = 0

def run_simulation(num_steps=200):
    """Run the simulation and return the collected data."""
    model = BoidModel(width=100, height=100, num_boids=50)
    for _ in range(num_steps):
        model.step()
    return model.datacollector

def plot_learning_metrics(data):
    """Plot various metrics to demonstrate learning progress."""
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Flock Behavior Metrics
    ax1 = fig.add_subplot(221)
    ax1.plot(data['Flock_Density'], label='Density')
    ax1.plot(data['Flock_Polarization'], label='Polarization')
    ax1.set_title('Flock Behavior Over Time')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Metric Value')
    ax1.legend()
    
    # Plot 2: Learning Progress
    ax2 = fig.add_subplot(222)
    ax2.plot(data['Average_Reward'], label='Avg Reward')
    ax2.plot(data['Average_Q_Value'], label='Avg Q-Value')
    ax2.set_title('Learning Progress')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Value')
    ax2.legend()
    
    # Plot 3: Exploration vs Exploitation
    ax3 = fig.add_subplot(223)
    ax3.plot(data['Exploration_Rate'], label='Exploration Rate')
    ax3.set_title('Exploration Rate Over Time')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Epsilon')
    ax3.legend()
    
    # Plot 4: Flocking Rules Balance
    ax4 = fig.add_subplot(224)
    ax4.plot(data['Separation'], label='Separation')
    ax4.plot(data['Alignment'], label='Alignment')
    ax4.plot(data['Cohesion'], label='Cohesion')
    ax4.set_title('Flocking Rules Balance')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Rule Strength')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

def plot_flock_state(model, step):
    """Plot the current state of the flock."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Get positions and velocities
    positions = np.array([agent.position for agent in model.schedule.agents])
    velocities = np.array([agent.velocity for agent in model.schedule.agents])
    
    # Plot positions
    ax.scatter(positions[:, 0], positions[:, 1], c='b', alpha=0.5)
    
    # Plot velocity vectors
    ax.quiver(positions[:, 0], positions[:, 1], 
              velocities[:, 0], velocities[:, 1], 
              color='r', alpha=0.3)
    
    ax.set_xlim(0, model.width)
    ax.set_ylim(0, model.height)
    ax.set_title(f'Flock State at Step {step}')
    plt.show()

if __name__ == "__main__":
    # Run simulation
    print("Running simulation...")
    data = run_simulation(200)
    
    # Get the model data as a pandas DataFrame
    model_data = data.get_model_vars_dataframe()
    
    # Plot learning metrics
    print("Plotting learning metrics...")
    plot_learning_metrics(model_data)
    
    # Create animation of flock movement (optional)
    print("Simulation complete. Check the plots to analyze the results.")

