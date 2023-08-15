#pip install pyglet==1.5.27
import numpy as np
import os
import sys
# Get the parent directory path
sys.path.append("..")
from gymnasium.utils import EzPickle
from pettingzoo.mpe._mpe_utils.core import Agent, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn
import random

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 4
        #world.damping = 1
        num_good_agents = 4
        num_adversaries = 6
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 1
        # num_food = 2
        num_forests = 2
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.leader = True if i == 0 else False
            agent.silent = True if i > 0 else False
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.075 if agent.adversary else 0.045
            agent.accel = 3.0 if agent.adversary else 4.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.45, 0.95, 0.45]) if not agent.adversary else np.array([0.95, 0.45, 0.45])
            agent.color -= np.array([0.3, 0.3, 0.3]) if agent.leader else np.array([0, 0, 0])
        
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            ###not implemented
            if agent.adversary == True:
                agent.weight = None
                agent.inventory = None
                agent.demand = random.randint(0, 50)
            else:
                agent.weight = 0
                agent.inventory = 50
                agent.demand = None

    def benchmark_data(self, agent, world):
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False


    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        #boundary_reward = -10 if self.outside_boundary(agent) else 0
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward


    def agent_reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        good_agent = self.good_agents(world)
        if shape:
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        
        if agent.collide:
            for a in good_agent:
                if self.is_collision(a, agent):
                    ### a.action not implemented ###
                    a.action = random.randint(0, 50)
                    agent.weight += a.action
                    a.weight += agent.weight

                    agent.inventory += agent.weight
                    a.inventory += a.weight
                    agent.weight = 0
                    a.weight = 0
                    rew += 0.1
    
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)  # 1 + (x - 1) * (x - 1)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= 2 * 0.1 * bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        rew = 0
        shape = True
        good_agent = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:
            rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in good_agent])
        # if agent.collide:
        #     for ag in agents:
        #         for adv in adversaries:
        #             if self.is_collision(ag, adv):
        #                 rew += 5
        for a in good_agent:
            if self.is_collision(a, agent):
                demand = agent.demand
                sales = min(demand, a.inventory)
                a.inventory -= sales
                rew += abs(demand - sales)
        return rew


    def observation2(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        food_pos = []
        for entity in world.food:
            if not entity.boundary:
                food_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        if not agent.adversary:
            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [agent.color] + entity_pos + entity_color + other_pos)
        else:
            #other_pos = list(reversed(other_pos)) if random.uniform(0,1) > 0.5 else other_pos  # randomize position of other agents in adversary network
            return np.concatenate([agent.state.p_vel] + entity_pos + other_pos)

if __name__ == '__main__':
    # parse arguments
    # load scenario from script
    # create world
    scenario = Scenario()
    world = scenario.make_world()
    # create multiagent environment
    from multiagent.environment import MultiAgentEnv
    from multiagent.policy import InteractivePolicy
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = False)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    policies = [InteractivePolicy(env,i) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()
    for i in range(1000):
        # query for action from each agent's policy
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)

        # render all agent views
        if i == 9999:
            env.render()

        # display rewards
        #for agent in env.world.agents:
        #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))

