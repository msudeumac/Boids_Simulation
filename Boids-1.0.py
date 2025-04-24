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
        # Initialize reward
        reward = 0

        # Get all boids within a certain radius
        nearby_boids = self.model.space.get_neighbors(self.position, 5, False)

        # Reward the boid for staying close to the flock
        if len(nearby_boids) > 0:
            reward += 1

        # Penalize the boid for collisions
        for other in nearby_boids:
            if np.linalg.norm(self.position - other.position) < 1:  # assuming a collision if distance is less than 1
                reward -= 5

        # Add your logic here to calculate reward based on state and action
        if action == 'separation':
            # If the boid is too close to others, reward it for taking the 'separation' action
            if np.linalg.norm(
                    self.position - state[0]) < 1:  # assuming the state is too close if distance is less than 1
                reward += 1
        elif action == 'alignment':
            # If the boid's velocity is different from the average, reward it for taking the 'alignment' action
            if np.linalg.norm(self.velocity - state[1]) > 1:  # assuming the state is misaligned if the difference in velocity is more than 1
                reward += 1
        elif action == 'cohesion':
            # If the boid is too far from others, reward it for taking the 'cohesion' action
            if np.linalg.norm(self.position - state[0]) > 5:  # assuming the state is too far if distance is more than 5
                reward += 1

        # Reward for maintaining a moderate speed
        speed = np.linalg.norm(self.velocity)
        if 1 < speed < 2:  # assuming moderate speed if speed is between 1 and 2
            reward += 1
        else:
            # Penalize for moving too fast or too slow
            reward -= 1

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
        self.schedule = RandomActivation(self) # Create a schedule and add it to the model
        self.q_values_over_time = []

        self.cohesion = 0
        self.alignment = 0
        self.separation = 0
        self.centering = 0


        # Create boids
        for i in range(self.num_boids):
            boid = BoidAgent(i, self)
            self.space.place_agent(boid, (self.random.uniform(0, self.width), self.random.uniform(0, self.height)))
            self.schedule.add(boid) # Add boid to the schedule

        self.datacollector = DataCollector(
            model_reporters={
                "AveragePosition": lambda m: np.mean([agent.position for agent in m.schedule.agents], axis=0),
                "AverageVelocity": lambda m: np.mean([agent.velocity for agent in m.schedule.agents], axis=0),
                "Cohesion": lambda m: m.cohesion,
                "Alignment": lambda m: m.alignment,
                "Separation": lambda m: m.separation,
                "Centering": lambda m: m.centering
            },
            agent_reporters={
                "QTable": lambda a: a.q_table  # New agent reporter
            }
        )

    def calculate_cohesion(self):
        distances = [np.linalg.norm(agent1.position - agent2.position)
                     for agent1 in self.schedule.agents
                     for agent2 in self.schedule.agents
                     if agent1 != agent2]
        self.cohesion = np.mean(distances)

    def calculate_alignment(self):
        velocities = [agent.velocity for agent in self.schedule.agents]
        self.alignment = np.var(velocities)

    def calculate_separation(self):
        distances = [np.linalg.norm(agent1.position - agent2.position)
                     for agent1 in self.schedule.agents
                     for agent2 in self.schedule.agents
                     if agent1 != agent2]
        self.separation = np.min(distances)

    def calculate_centering(self):
        positions = [agent.position for agent in self.schedule.agents]
        self.centering = np.var(positions)

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)
        self.calculate_cohesion()
        self.calculate_alignment()
        self.calculate_separation()
        self.calculate_centering()
        self.q_values_over_time.append({agent.unique_id: agent.q_table for agent in self.schedule.agents})
        self.states = set().union(*(agent.states for agent in self.schedule.agents))


model = BoidModel(width=100, height=100, num_boids=50)

for i in range(200):  # Adjust the number of steps as needed
    model.step()


# Get the collected data
data = model.datacollector.get_model_vars_dataframe()

# Plot the measures over time
plt.figure(figsize=(10, 10))
plt.plot(data['Cohesion'], label='Cohesion')
plt.plot(data['Alignment'], label='Alignment')
plt.plot(data['Separation'], label='Separation')
plt.plot(data['Centering'], label='Centering')
plt.title('Flock Stability Measures Over Time')
plt.xlabel('Time Step')
plt.ylabel('Measure')
plt.legend()
plt.show()









# Define the number of episodes
num_episodes = 200

# Initialize an empty list to store the average Q-values for each episode
average_q_values_per_episode = []

# Run the model for a certain number of episodes
for episode in range(num_episodes):
    # Initialize the model
    model = BoidModel(width=100, height=100, num_boids=50)

    # Run the model for a certain number of steps
    for i in range(50):
        model.step()

    # Initialize an empty list to store the Q-values for the current episode
    current_q_values = []

    # Iterate over all agents in the model's schedule
    for agent in model.schedule.agents:
        # Get all the Q-values for the current agent
        q_values = agent.q_table

        # For each state-action pair, add the Q-value to the list
        for state, actions in q_values.items():
            for action, q_value in actions.items():
                current_q_values.append(q_value)

    # Add the average Q-value for the current episode to the list
    average_q_values_per_episode.append(sum(current_q_values) / len(current_q_values) if current_q_values else 0)

# Plot the average Q-values per episode
plt.plot(average_q_values_per_episode)

# Add a title and labels
plt.title('Average Q-values Per Episode')
plt.xlabel('Episode')
plt.ylabel('Average Q-value')

# Show the plot
plt.show()






##################
model = BoidModel(width=100, height=100, num_boids=50)

# Run the model for a certain number of steps
for i in range(50):
    model.step()






# Initialize an empty list to store the Q-values
all_q_values = []

# Iterate over all agents in the model's schedule
for agent in model.schedule.agents:
    # Get all the Q-values for the current agent
    q_values = agent.q_table

    # For each state-action pair, add the Q-value to the list
    for state, actions in q_values.items():
        for action, q_value in actions.items():
            all_q_values.append(q_value)

# Create a histogram of the Q-values
plt.hist(all_q_values, bins=50, edgecolor='black')

# Add a title and labels
plt.title('Distribution of Q-values')
plt.xlabel('Q-value')
plt.ylabel('Frequency')

# Show the plot
plt.show()




# Initialize an empty list to store the Q-values at each time step
q_values_over_time = []

# Run the model for a certain number of steps
for i in range(50):
    model.step()

    # Initialize an empty list to store the Q-values for the current time step
    current_q_values = []

    # Iterate over all agents in the model's schedule
    for agent in model.schedule.agents:
        # Get all the Q-values for the current agent
        q_values = agent.q_table

        # For each state-action pair, add the Q-value to the list
        for state, actions in q_values.items():
            for action, q_value in actions.items():
                current_q_values.append(q_value)

    # Add the average Q-value for the current time step to the list
    q_values_over_time.append(sum(current_q_values) / len(current_q_values) if current_q_values else 0)

# Plot the Q-values over time
plt.plot(q_values_over_time)

# Add a title and labels
plt.title('Average Q-values Over Time')
plt.xlabel('Time Step')
plt.ylabel('Average Q-value')

# Show the plot
plt.show()

