import importlib.util
import sys
import os
import yaml
from environments.collector.wrappers import CollectorGymEnv, RecordEpisode
from environments.collector.params import EnvParams
import argparse

def load_agent(agent_dir):
    """
    Loads an Agent instance from a directory containing agent.py and config.yaml
    """
    # Load config
    config_path = os.path.join(agent_dir, "config.yaml")
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Optionally convert to an object or pass as-is
    class Config: pass
    config = Config()
    for k, v in config_dict.items():
        setattr(config, k, v)

    # Load Agent class from agent.py
    agent_file = os.path.join(agent_dir, "agent.py")
    spec = importlib.util.spec_from_file_location("agent", agent_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.Agent(config)

def compete(agent1_dir, agent2_dir, seed=0, max_steps=1000, output_path=None):
    agent0 = load_agent(agent1_dir)
    agent0.load()
    agent1 = load_agent(agent2_dir)
    agent1.load()

    env = CollectorGymEnv(numpy_output=True)
    env = RecordEpisode(env,save_on_reset=False, save_on_close=False) if output_path else env
    env_params = EnvParams()
    obs, info = env.reset(options=dict(params=env_params))

 
    done = False
    total_reward0 = 0
    total_reward1 = 0
    steps = 0

    while not done and steps < max_steps:
        action0 = agent0.act(obs["player_0"])
        action1 = agent1.act(obs["player_1"])
        actions = {"player_0": action0, "player_1": action1}

        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward0 += reward[0]
        total_reward1 += reward[1]
        done = terminated or truncated
        steps += 1

    print(f"\nGame Over in {steps} steps")
    print(f"Agent 0 ({agent1_dir}) reward: {total_reward0}")
    print(f"Agent 1 ({agent2_dir}) reward: {total_reward1}")
    print(f"Final score: {info['state'].team_points[0]} - {info['state'].team_points[1]}")
    if output_path:
        env.save_episode(output_path)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("agent1")
    parser.add_argument("agent2")
    parser.add_argument("--output", type=str, default=None, help="Path to save replay (e.g. replay.json)")
    args = parser.parse_args()
    compete(args.agent1, args.agent2, output_path=args.output)
