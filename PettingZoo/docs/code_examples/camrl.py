from pettingzoo.mpe import camrl

env = camrl.env(render_mode="human")
env.reset(seed=42)
import pdb
pdb.set_trace()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()

    env.step(action)
env.close()
