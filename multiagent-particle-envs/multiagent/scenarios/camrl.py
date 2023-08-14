import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
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
            rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in agents])
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
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # in_forest = [np.array([-1]), np.array([-1])]
        # inf1 = False
        # inf2 = False
        # if self.is_collision(agent, world.forests[0]):
        #     in_forest[0] = np.array([1])
        #     inf1= True
        # if self.is_collision(agent, world.forests[1]):
        #     in_forest[1] = np.array([1])
        #     inf2 = True

        food_pos = []
        for entity in world.food:
            if not entity.boundary:
                food_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        # other_pos = []
        # other_vel = []
        # for other in world.agents:
        #     if other is agent: continue
        #     comm.append(other.state.c)
        #     oth_f1 = self.is_collision(other, world.forests[0])
        #     oth_f2 = self.is_collision(other, world.forests[1])
        #     if (inf1 and oth_f1) or (inf2 and oth_f2) or (not inf1 and not oth_f1 and not inf2 and not oth_f2) or agent.leader:  #without forest vis
        #         other_pos.append(other.state.p_pos - agent.state.p_pos)
        #         if not other.adversary:
        #             other_vel.append(other.state.p_vel)
        #     else:
        #         other_pos.append([0, 0])
        #         if not other.adversary:
        #             other_vel.append([0, 0])

        # to tell the pred when the prey are in the forest
        # prey_forest = []
        # ga = self.good_agents(world)
        # for a in ga:
        #     if any([self.is_collision(a, f) for f in world.forests]):
        #         prey_forest.append(np.array([1]))
        #     else:
        #         prey_forest.append(np.array([-1]))
        # # to tell leader when pred are in forest
        # prey_forest_lead = []
        # for f in world.forests:
        #     if any([self.is_collision(a, f) for a in ga]):
        #         prey_forest_lead.append(np.array([1]))
        #     else:
        #         prey_forest_lead.append(np.array([-1]))

        comm = [world.agents[0].state.c]

        if agent.adversary and not agent.leader:
            # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel + in_forest + comm)
            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + comm)
        if agent.leader:
            return np.concatenate(
                # [agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel + in_forest + comm)
                [agent.state.p_vel] + [agent.state.p_pos] + entity_pos + comm)
        else:
            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos)

        # def observation(self, agent, world):
        #     # goal color
        #     goal_color = [np.zeros(world.dim_color), np.zeros(world.dim_color)]
        #     if agent.goal_b is not None:
        #         goal_color[1] = agent.goal_b.color 

        #     # get positions of all entities in this agent's reference frame
        #     entity_pos = []
        #     for entity in world.landmarks:
        #         entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        #     # entity colors
        #     entity_color = []
        #     for entity in world.landmarks:
        #         entity_color.append(entity.color)
        #     # communication of all other agents
        #     comm = []
        #     for other in world.agents:
        #         if other is agent: continue
        #         comm.append(other.state.c)
        #     return np.concatenate([agent.state.p_vel] + entity_pos + [goal_color[1]] + comm)
                