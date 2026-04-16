import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import jax.numpy as jnp
from environments.collector.wrappers import CollectorGymEnv
from environments.collector.params import EnvParams
from agents.baseline.agent import Agent
import sys
import yaml
import os
class HumanCollectorUI:
    def __init__(self, seed=0):
        self.env = CollectorGymEnv(numpy_output=True)
        self.env_params = EnvParams()
        self.obs, info = self.env.reset(seed=seed, options=dict(params=self.env_params))
        self.state = info['state']
        self.step = 0
        self.total_reward = 0
        self.opponent_total_reward = 0
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        plt.subplots_adjust(bottom=0.15)
        config_path = os.path.join("src/agents/baseline", "config.yaml")
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        class Config: pass
        config = Config()
        for k, v in config_dict.items():
            setattr(config, k, v)
        self.agent = Agent(config)
        

        self.cmap = mcolors.LinearSegmentedColormap.from_list(
            "landscape", ["green", "#228B22", "darkgreen", "#8B4513", "gray", "#A9A9A9", "whitesmoke"], N=256
        )

        # Init map 
        tile_map = jnp.array(self.state.map_features.tile_type)
        
        self.im = self.ax.imshow(tile_map, cmap=self.cmap, vmin=0, vmax=2)
        

        # Agent marker
        self.agent_marker, = self.ax.plot([], [], "ro", markersize=8)
        self.opponent_marker, = self.ax.plot([], [], "bo", markersize=8)
        self.text = self.ax.text(0.02, 0.95, f"Step: 0", transform=self.ax.transAxes, color="white", fontsize=12)

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.update_display()
        plt.show()

    def update_display(self):
        tile_map = jnp.array(self.state.map_features.tile_type)
      
        self.im.set_array(tile_map)

        agent_y, agent_x = self.state.units.position[0]
        opponent_y, opponent_x = self.state.units.position[1]
        self.agent_marker.set_data([agent_x], [agent_y])
        self.opponent_marker.set_data([opponent_x], [opponent_y])
        self.text.set_text(f"Step: {self.step} | Items: {self.state.team_points[0]} - {self.state.team_points[1]}  | Reward: {self.total_reward} - {self.opponent_total_reward}")
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        key_to_action = {
            "up": 0,
            "right": 1,
            "down": 2,
            "left": 3,
        }

        if event.key in key_to_action:
            action = key_to_action[event.key]
            opponent_action = self.agent.act(self.obs["player_1"]) #self.env.action_space.sample()
            actions = {"player_0": action , "player_1": opponent_action }
            #print(self.obs["player_1"])
            next_obs, reward, _, _, info = self.env.step(actions)
            self.obs = next_obs
            self.state = info['state']
            self.total_reward += reward[0]
            self.opponent_total_reward += reward[1]
            self.step += 1
            self.update_display()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python play_collector.py <seed> seed set to 0")
        seed = 0
    else:
        seed = int(sys.argv[1])
    ui = HumanCollectorUI(seed=seed)
