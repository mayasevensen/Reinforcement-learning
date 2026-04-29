import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from types import SimpleNamespace
from agents.agent_base import BaseAgent
from environments.collector.state import EnvState


class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


def preprocess_obs(obs):
    """
    Smart feature engineering instead of raw map.
    Much smaller input = faster and easier to learn.
    """
    raw_map = obs['map_features']['tile_type']
    my_pos = obs['units']['position'][0].astype(np.float32)
    opp_pos = obs['units']['position'][1].astype(np.float32)
    W, H = raw_map.shape

    # --- Positional features ---
    my_pos_norm = my_pos / np.array([W, H], dtype=np.float32)
    opp_pos_norm = opp_pos / np.array([W, H], dtype=np.float32)

    # --- Relative opponent position ---
    rel_opp = (opp_pos - my_pos) / np.array([W, H], dtype=np.float32)

    # --- Nearest items: directions + distances to top-5 closest items ---
    item_locs = np.argwhere(raw_map == 2).astype(np.float32)
    item_features = np.zeros(5 * 3, dtype=np.float32)  # 5 items x (dx, dy, dist)
    if len(item_locs) > 0:
        diffs = item_locs - my_pos
        dists = np.abs(diffs).sum(axis=1)
        order = np.argsort(dists)[:5]
        for i, idx in enumerate(order):
            dx = diffs[idx][0] / W
            dy = diffs[idx][1] / H
            d = dists[idx] / (W + H)
            item_features[i*3:(i+1)*3] = [dx, dy, d]

    # --- Direction to single nearest item (explicit) ---
    if len(item_locs) > 0:
        dists = np.abs(item_locs - my_pos).sum(axis=1)
        nearest = item_locs[np.argmin(dists)]
        direction = (nearest - my_pos).astype(np.float32)
        dist_to_nearest = np.abs(direction).sum() / (W + H)
        direction = direction / (np.abs(direction).sum() + 1e-8)
    else:
        direction = np.zeros(2, dtype=np.float32)
        dist_to_nearest = np.array([1.0], dtype=np.float32)

    # --- Local 5x5 window around agent (obstacles + items nearby) ---
    radius = 2
    local = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            ni, nj = int(my_pos[0]) + di, int(my_pos[1]) + dj
            if 0 <= ni < W and 0 <= nj < H:
                local[di + radius, dj + radius] = raw_map[ni, nj] / 2.0
            else:
                local[di + radius, dj + radius] = 0.5  # treat out-of-bounds as wall

    # --- Global stats ---
    items_on_map = np.array([obs['items_on_map'].item() / 50.0], dtype=np.float32)
    steps_norm = np.array([obs['steps'].item() / 1000.0], dtype=np.float32)
    team_points = obs['team_points'].astype(np.float32).flatten() / 50.0

    # --- Danger: is opponent adjacent? ---
    opp_dist = np.abs(opp_pos - my_pos).sum()
    opp_close = np.array([1.0 if opp_dist <= 2 else 0.0], dtype=np.float32)

    return np.concatenate([
        my_pos_norm,          # 2
        opp_pos_norm,         # 2
        rel_opp,              # 2
        direction,            # 2
        [dist_to_nearest],    # 1
        item_features,        # 15
        local.flatten(),      # 25
        team_points,          # 2
        items_on_map,         # 1
        steps_norm,           # 1
        opp_close,            # 1
    ])                        # total: 54


class Agent(BaseAgent):
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.epsilon = getattr(config, 'epsilon_start', 1.0)
        self.epsilon_end = getattr(config, 'epsilon_end', 0.05)
        self.epsilon_decay = getattr(config, 'epsilon_decay', 0.998)
        self.gamma = getattr(config, 'gamma', 0.99)
        self.batch_size = getattr(config, 'batch_size', 128)
        self.target_update_freq = getattr(config, 'target_update_freq', 100)
        self.training = getattr(config, 'training', False)
        self.hidden_dim = getattr(config, 'hidden_dim', 128)
        self.lr = getattr(config, 'learning_rate', 0.001)

        self.q_net = None
        self.target_net = None
        self.optimizer = None

        self.replay_buffer = ReplayBuffer(getattr(config, 'buffer_size', 20000))
        self.min_buffer_size = getattr(config, 'min_buffer_size', 200)

        self._step_count = 0
        self._last_state = None
        self._last_action = None

    def _build_networks(self, input_dim):
        self.input_dim = input_dim
        self.q_net = QNetwork(input_dim, self.hidden_dim).to(self.device)
        self.target_net = QNetwork(input_dim, self.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)

    def act(self, observation: EnvState) -> int:
        state = preprocess_obs(observation)

        if self.q_net is None:
            self._build_networks(len(state))
            if hasattr(self, '_pending_load_path'):
                self.q_net.load_state_dict(
                    torch.load(self._pending_load_path, map_location=self.device)
                )
                self.target_net.load_state_dict(self.q_net.state_dict())
                if not self.training:
                    self.q_net.eval()
                del self._pending_load_path
                print("Weights loaded!")

        if self.training and random.random() < self.epsilon:
            action = random.randint(0, 3)
        else:
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action = self.q_net(s).argmax(dim=1).item()

        self._last_state = state
        self._last_action = action
        return action

    def store(self, next_obs, reward, done):
        if not self.training or self._last_state is None:
            return
        next_state = preprocess_obs(next_obs)
        self.replay_buffer.push(self._last_state, self._last_action, reward, next_state, float(done))
        self._step_count += 1
        if self._step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def train_step(self):
        if not self.training or len(self.replay_buffer) < self.min_buffer_size:
            return
        self._train_step()

    def update(self, next_obs, reward, done):
        self.store(next_obs, reward, done)
        self.train_step()

    def _train_step(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

    def end_episode(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path=None):
        save_path = path or self.config.weights_dir
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.q_net.state_dict(), os.path.join(save_path, "weights.pth"))
        print(f"Saved weights to {save_path}")

    def load(self) -> None:
        weights_path = os.path.join(self.config.weights_dir, "weights.pth")
        if not os.path.exists(weights_path):
            print(f"No weights found at {weights_path} — starting fresh.")
            return
        self._pending_load_path = weights_path
        print(f"Weights queued for loading from {weights_path}")