import gym
from gym import spaces
import numpy as np
import networkx as nx
import random

# Create a directed graph G using NetworkX to represent the supply chain
G = nx.DiGraph()
G.add_node('Supplier') # Add a Supplier node
G.add_node('Retailer') # Add a Retailer node
G.add_edge('Supplier', 'Retailer', weight=0)
 # Add an edge from the Supplier to the Retailer with an initial weight of 0

# Define an Agent class
class Agent:
    def __init__(self, name):
        self.name = name # Name of the agent

    # Define a random action the agent can take
    def act(self, state):
        return random.randint(0, 50) # Random integer between 0 and 50

# Define a custom Gym Environment
class SupplyChainEnv(gym.Env):
    def __init__(self, G, agents):
        super(SupplyChainEnv, self).__init__()
        self.G = G # Supply chain graph
        self.agents = agents # Agents in the supply chain
        
        # Define the action space as multi-discrete, where each agent can perform an action from 0 to 50
        self.action_space = spaces.MultiDiscrete([51 for _ in range(len(self.agents))])

        # Define the observation space as a box from 0 to 50 with size equal to the number of nodes in the graph
        self.observation_space = spaces.Box(low=0, high=50, shape=(len(self.G.nodes),))

        self.reset() # Reset the environment

    # Define the step function which will execute the agent's action and return the new state, reward, done status, and extra info
    def step(self, action):
        total_reward = 0 # Initialize total reward
        
        # Update graph based on actions
        for i, (node, agent) in enumerate(self.agents.items()):
            if self.G.in_edges(node): # If the node has incoming edges
                supplier, _ = list(self.G.in_edges(node))[0] # Get the supplier for the node
                self.G[supplier][node]['weight'] += action[i] # Update the weight of the edge based on the action

        # Update inventory and calculate rewards
        total_reward = 0
        for node in self.G.nodes:
            if self.G.in_edges(node):  # If the node has incoming edges
                supplier, _ = list(self.G.in_edges(node))[0] # Get the supplier for the node
                shipment = self.G[supplier][node]['weight']  # Get the shipment from the supplier
                self.G.nodes[node]['inventory'] += shipment  # Add the shipment to the inventory
                self.G[supplier][node]['weight'] = 0 # Reset the weight of the edge

            # Calculate demand, sales, and update inventory
            demand = self.G.nodes[node]['demand']
            sales = min(demand, self.G.nodes[node]['inventory'])
            self.G.nodes[node]['inventory'] -= sales

            # Calculate reward as the negative absolute difference between demand and sales
            reward = -abs(demand - sales)
            total_reward += reward # Add to the total reward

        return np.array([self.G.nodes[node]['inventory'] for node in self.G.nodes]).astype(float), total_reward, False, {}

    # Define the reset function to reset the environment to its initial state
    def reset(self):
        for node in self.G.nodes:
            self.G.nodes[node]['inventory'] = 50 # Reset inventory to 50
            self.G.nodes[node]['demand'] = random.randint(0, 50) # Set a random demand between 0 and 50
        return np.array([self.G.nodes[node]['inventory'] for node in self.G.nodes]).astype(float) # Return the initial state

# Create agents
agents = {node: Agent(node) for node in G.nodes}  # Create an agent for each node in the graph

# Create environment
env = SupplyChainEnv(G, agents)  # Create the supply chain environment with the graph and agents

# Import the PPO algorithm from Stable Baselines 3
from stable_baselines3 import PPO

# Initialize a PPO model with an MLP policy and the custom environment
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)


