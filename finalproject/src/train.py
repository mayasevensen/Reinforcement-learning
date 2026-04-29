import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import numpy as np
from environments.collector.wrappers import CollectorGymEnv
from environments.collector.params import EnvParams

with open("src/agents/agent/config.yaml") as f:
    config_dict = yaml.safe_load(f)

class Config: pass
config = Config()
for k, v in config_dict.items():
    setattr(config, k, v)

config.training = True
config.epsilon_start = 1.0
config.epsilon_end = 0.05
config.epsilon_decay = 0.998

sys.path.insert(0, "src/agents/agent")
from agent import Agent

agent = Agent(config)
agent.load()

env = CollectorGymEnv(numpy_output=True)
env_params = EnvParams()

NUM_EPISODES = 3000
MAX_STEPS = 500
SAVE_EVERY = 500
PRINT_EVERY = 50
TRAIN_EVERY = getattr(config, 'train_every', 4)

episode_rewards = []
episode_scores = []


def get_nearest_item_dist(obs):
    raw_map = obs['map_features']['tile_type']
    my_pos = obs['units']['position'][0]
    item_locs = np.argwhere(raw_map == 2)
    if len(item_locs) == 0:
        return None
    dists = np.abs(item_locs - my_pos).sum(axis=1)
    return float(dists.min())


def shape_reward(obs, next_obs, raw_reward):
    shaped = raw_reward
    if raw_reward > 0:        # collected item
        shaped += 2.0
    if raw_reward == -2.0:    # hit wall
        shaped -= 1.0
    dist_before = get_nearest_item_dist(obs)
    dist_after = get_nearest_item_dist(next_obs)
    if dist_before is not None and dist_after is not None:
        delta = dist_before - dist_after  # positive = got closer
        shaped += 0.3 * delta
    return shaped


for episode in range(NUM_EPISODES):
    obs, info = env.reset(options=dict(params=env_params))
    total_reward = 0
    done = False
    steps = 0

    while not done and steps < MAX_STEPS:
        action = agent.act(obs["player_0"])
        opp_action = env.action_space.sample()
        actions = {"player_0": action, "player_1": opp_action}

        next_obs, reward, terminated, truncated, info = env.step(actions)

        raw_r = float(reward[0])
        shaped_r = shape_reward(obs["player_0"], next_obs["player_0"], raw_r)

        done = terminated or truncated
        agent.store(next_obs["player_0"], shaped_r, done)
        if steps % TRAIN_EVERY == 0:
            agent.train_step()

        obs = next_obs
        total_reward += raw_r
        steps += 1

    episode_rewards.append(total_reward)
    episode_scores.append(int(info['state'].team_points[0]))
    agent.end_episode()

    if (episode + 1) % PRINT_EVERY == 0:
        avg_r = np.mean(episode_rewards[-PRINT_EVERY:])
        avg_s = np.mean(episode_scores[-PRINT_EVERY:])
        print(f"Episode {episode+1:5d} | Avg Reward: {avg_r:7.1f} | "
              f"Avg Score: {avg_s:.2f} | Epsilon: {agent.epsilon:.3f} | "
              f"Buffer: {len(agent.replay_buffer)}")

    if (episode + 1) % SAVE_EVERY == 0:
        agent.save()

agent.save()
print("Training complete!")
env.close()