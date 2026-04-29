import json
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Button
import jax.numpy as jnp

class EpisodeViewer:
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.data = json.load(f)

        
        self.frames = self.data["observations"]
        self.actions = self.data.get("actions", [])
        self.rewards = self.data.get("rewards", [])
        self.current_step = 0
        self.paused = True
        
        self.total_reward = 0
        self.opponent_total_reward = 0

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        plt.subplots_adjust(bottom=0.2)

        self.frame_interval = 500  # adjust speed here (ms)
        self.timer = self.fig.canvas.new_timer(interval=self.frame_interval)
        self.timer.add_callback(self.play_loop)

        self.cmap = mcolors.LinearSegmentedColormap.from_list(
            "landscape", ["green", "#228B22", "darkgreen", "#8B4513", "gray", "#A9A9A9", "whitesmoke"], N=256
        )
        self.fog_color = jnp.array([0.5, 0.5, 0.5, 0.5])  # RGBA gray

        # Initialize plot with the first frame
        self.im = self.ax.imshow(jnp.array(self.frames[0]["map_features"]["tile_type"]), cmap=self.cmap)
        
        self.agent_marker, = self.ax.plot([], [], "ro", markersize=8)  # Red dot for agent
        self.opponent_marker, = self.ax.plot([], [], "bo", markersize=8)
        self.text = self.ax.text(0.02, 0.95, f"Step: 0", transform=self.ax.transAxes, color="white", fontsize=12)

        # Buttons
        self.add_buttons()

        # Key bindings
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.update_frame()
        plt.show()

    def add_buttons(self):
        axprev = plt.axes([0.1, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.21, 0.05, 0.1, 0.075])
        axplay = plt.axes([0.32, 0.05, 0.1, 0.075])

        self.bprev = Button(axprev, "◀️ Prev")
        self.bnext = Button(axnext, "Next ▶️")
        self.bplay = Button(axplay, "▶️ Play")

        self.bprev.on_clicked(self.prev_frame)
        self.bnext.on_clicked(self.next_frame)
        self.bplay.on_clicked(self.toggle_pause)

    def on_key(self, event):
        if event.key == "right":
            self.next_frame(None)
        elif event.key == "left":
            self.prev_frame(None)
        elif event.key == " ":
            self.toggle_pause(None)

    def update_frame(self):
        frame = self.frames[self.current_step]
        reward = self.rewards[self.current_step-1] if self.current_step>0 else [0,0]
        # Update terrain visualization
        terrain = jnp.array(frame["map_features"]["tile_type"])
        self.im.set_array(terrain)
        
        
  
        # Update agent position
        agent_pos = frame["units"]["position"][0]
        opponent_pos = frame["units"]["position"][1]
        if isinstance(agent_pos, list) and len(agent_pos) == 2:
            self.agent_marker.set_data([agent_pos[1]], [agent_pos[0]])  # Ensure x, y are sequences
        else:
            self.agent_marker.set_data(agent_pos[1], agent_pos[0])  # y,x order
        if isinstance(opponent_pos, list) and len(opponent_pos) == 2:
            self.opponent_marker.set_data([opponent_pos[1]], [opponent_pos[0]])
        else:
            self.opponent_marker.set_data(opponent_pos[1], opponent_pos[0])

        self.total_reward += reward[0]
        self.opponent_total_reward += reward[1]
        # Update step counter
        self.text.set_text(f"Step: {self.current_step} | Items: {frame['team_points'][0]} - {frame['team_points'][1]}")

        self.fig.canvas.draw_idle()

    def next_frame(self, event):
        self.paused = True  # Pause the game if user manually steps
        self.bplay.label.set_text("▶️ Play")  # Update button text
        if self.current_step < len(self.frames) - 1:
            self.current_step += 1
            self.update_frame()

    def prev_frame(self, event):
        self.paused = True  # Pause the game if user manually steps
        self.bplay.label.set_text("▶️ Play")  # Update button text
        if self.current_step > 0:
            self.current_step -= 1
            self.update_frame()

    def toggle_pause(self, event):
        self.paused = not self.paused
        self.bplay.label.set_text("II Pause" if not self.paused else "▶️ Play")
        if not self.paused:
            self.timer.start()
        else:
            self.timer.stop()

    def play_loop(self):
        if not self.paused and self.current_step < len(self.frames) - 1:
            self.current_step += 1
            self.update_frame()
        else:
            self.paused = True
            self.bplay.label.set_text("▶️ Play")
            self.timer.stop()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualiser.py <path_to_json_file>")
        sys.exit(1)

    json_path = sys.argv[1]  # Get the filename from the command line
    viewer = EpisodeViewer(json_path)
    