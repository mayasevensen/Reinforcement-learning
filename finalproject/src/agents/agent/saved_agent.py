"""
DQN agent for the Collector environment.

Improvements over the previous version:
  - Dueling DQN head (separate V(s) and A(s,a) streams).
  - Huber loss instead of MSE for more stable Q-learning.
  - BFS distance to nearest reachable item (handles obstacles, unlike Manhattan).
  - "Opponent closer to my target" feature -- info baseline cannot act on.
  - Cleaner episode-boundary handling for the replay buffer.
  - Optional opponent-aware reward shaping is done in train.py, not here,
    so this file stays usable for both training and inference.
"""

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

import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


# ---------------------------------------------------------------------------
# Network: Dueling DQN
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    """
    Dueling architecture: shared trunk -> two heads (value + advantage).
    Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))

    Why dueling: in this env, many states have similar value regardless of
    action (e.g. when no item is adjacent and you're moving toward one).
    Separating value from advantage lets the network learn V(s) from every
    transition, even ones where the action choice barely matters.
    """

    def __init__(self, input_dim, hidden_dim, output_dim=4):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        h = self.trunk(x)
        v = self.value_head(h)                    # (B, 1)
        a = self.advantage_head(h)                # (B, 4)
        # Subtract mean advantage to make V/A identifiable.
        return v + (a - a.mean(dim=1, keepdim=True))


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
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
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def bfs_distances(tile_map, start):
    """
    BFS from `start` over walkable tiles (tile_type != 1).
    Returns an int array of shape (H, W) where unreachable cells are -1.
    Cheap on 16x16.
    """
    H, W = tile_map.shape
    dist = -np.ones((H, W), dtype=np.int32)
    sy, sx = int(start[0]), int(start[1])
    if not (0 <= sy < H and 0 <= sx < W) or tile_map[sy, sx] == 1:
        return dist
    dist[sy, sx] = 0
    q = deque([(sy, sx)])
    while q:
        y, x = q.popleft()
        d = dist[y, x]
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and dist[ny, nx] == -1 and tile_map[ny, nx] != 1:
                dist[ny, nx] = d + 1
                q.append((ny, nx))
    return dist


def preprocess_obs(obs):
    """
    Build a fixed-size feature vector. Every component is in roughly [-1, 1].

    Layout (total 56 dims):
        my_pos_norm                   2
        opp_pos_norm                  2
        rel_opp                       2
        direction_to_nearest          2
        dist_to_nearest               1
        item_features (top 5)        15   (dx, dy, dist) per item
        local 5x5 obstacles + items  25
        team_points (mine, opp)       2
        items_on_map                  1
        steps_norm                    1
        opp_close_flag                1
        opp_closer_to_my_target       1   <-- key competitive feature
        bfs_dist_to_nearest_item      1   <-- handles obstacles
    """
    raw_map = obs['map_features']['tile_type']
    my_pos = obs['units']['position'][0].astype(np.float32)
    opp_pos = obs['units']['position'][1].astype(np.float32)
    H, W = raw_map.shape
    diag = float(W + H)

    my_pos_norm = my_pos / np.array([H, W], dtype=np.float32)
    opp_pos_norm = opp_pos / np.array([H, W], dtype=np.float32)
    rel_opp = (opp_pos - my_pos) / np.array([H, W], dtype=np.float32)

    item_locs = np.argwhere(raw_map == 2).astype(np.float32)

    # Top-5 nearest items (by Manhattan, fast)
    item_features = np.zeros(5 * 3, dtype=np.float32)
    nearest = None
    if len(item_locs) > 0:
        diffs = item_locs - my_pos
        dists = np.abs(diffs).sum(axis=1)
        order = np.argsort(dists)[:5]
        for i, idx in enumerate(order):
            item_features[i * 3] = diffs[idx][0] / H
            item_features[i * 3 + 1] = diffs[idx][1] / W
            item_features[i * 3 + 2] = dists[idx] / diag
        nearest = item_locs[order[0]]

    # Direction + Manhattan dist to nearest
    if nearest is not None:
        dvec = nearest - my_pos
        dist_to_nearest_man = float(np.abs(dvec).sum() / diag)
        norm = np.abs(dvec).sum() + 1e-8
        direction = (dvec / norm).astype(np.float32)
    else:
        direction = np.zeros(2, dtype=np.float32)
        dist_to_nearest_man = 1.0

    # Local 5x5 window (out-of-bounds treated as obstacle)
    radius = 2
    local = np.full((2 * radius + 1, 2 * radius + 1), 0.5, dtype=np.float32)
    my_y, my_x = int(my_pos[0]), int(my_pos[1])
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            ni, nj = my_y + di, my_x + dj
            if 0 <= ni < H and 0 <= nj < W:
                local[di + radius, dj + radius] = raw_map[ni, nj] / 2.0

    items_on_map = np.array([obs['items_on_map'].item() / 50.0], dtype=np.float32)
    steps_norm = np.array([obs['steps'].item() / 1000.0], dtype=np.float32)
    team_points = obs['team_points'].astype(np.float32).flatten() / 50.0

    opp_dist_man = float(np.abs(opp_pos - my_pos).sum())
    opp_close = np.array([1.0 if opp_dist_man <= 2 else 0.0], dtype=np.float32)

    # Competitive feature: is opponent closer (Manhattan) to my nearest item?
    if nearest is not None:
        my_d = float(np.abs(nearest - my_pos).sum())
        opp_d = float(np.abs(nearest - opp_pos).sum())
        opp_closer = np.array([1.0 if opp_d < my_d else 0.0], dtype=np.float32)
    else:
        opp_closer = np.array([0.0], dtype=np.float32)

    # BFS distance to nearest reachable item (true graph distance)
    if len(item_locs) > 0:
        dmap = bfs_distances(raw_map, my_pos)
        item_locs_int = item_locs.astype(np.int32)
        ds = dmap[item_locs_int[:, 0], item_locs_int[:, 1]]
        reachable = ds[ds >= 0]
        if reachable.size > 0:
            bfs_d = np.array([reachable.min() / diag], dtype=np.float32)
        else:
            bfs_d = np.array([1.0], dtype=np.float32)
    else:
        bfs_d = np.array([1.0], dtype=np.float32)

    return np.concatenate([
        my_pos_norm,                              # 2
        opp_pos_norm,                             # 2
        rel_opp,                                  # 2
        direction,                                # 2
        np.array([dist_to_nearest_man], dtype=np.float32),  # 1
        item_features,                            # 15
        local.flatten(),                          # 25
        team_points,                              # 2
        items_on_map,                             # 1
        steps_norm,                               # 1
        opp_close,                                # 1
        opp_closer,                               # 1
        bfs_d,                                    # 1
    ])                                            # total: 56


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
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
        self.target_update_freq = getattr(config, 'target_update_freq', 500)
        self.training = getattr(config, 'training', False)
        self.hidden_dim = getattr(config, 'hidden_dim', 128)
        self.lr = getattr(config, 'learning_rate', 0.0005)

        self.q_net = None
        self.target_net = None
        self.optimizer = None
        self.input_dim = None

        self.replay_buffer = ReplayBuffer(getattr(config, 'buffer_size', 50000))
        self.min_buffer_size = getattr(config, 'min_buffer_size', 1000)

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

    def reset_episode(self):
        """Call at the start of every episode to prevent buffer transitions
        from crossing episode boundaries."""
        self._last_state = None
        self._last_action = None

    def act(self, observation: EnvState) -> int:
        state = preprocess_obs(observation)

        if self.q_net is None:
            self._build_networks(len(state))
            if hasattr(self, '_pending_load_path'):
                try:
                    sd = torch.load(self._pending_load_path, map_location=self.device)
                    self.q_net.load_state_dict(sd)
                    self.target_net.load_state_dict(self.q_net.state_dict())
                    if not self.training:
                        self.q_net.eval()
                    print("Weights loaded!")
                except Exception as e:
                    # Architecture mismatch (e.g. loading old non-dueling weights):
                    # start fresh rather than crash.
                    print(f"Could not load weights ({e}); starting fresh.")
                del self._pending_load_path

        if self.training and random.random() < self.epsilon:
            action = random.randint(0, 3)
        else:
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action = int(self.q_net(s).argmax(dim=1).item())

        self._last_state = state
        self._last_action = action
        return action

    def store(self, next_obs, reward, done):
        if not self.training or self._last_state is None:
            return
        next_state = preprocess_obs(next_obs)
        self.replay_buffer.push(
            self._last_state, self._last_action, reward, next_state, float(done)
        )
        self._step_count += 1
        if self._step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def train_step(self):
        if not self.training or len(self.replay_buffer) < self.min_buffer_size:
            return None
        return self._train_step()

    def update(self, next_obs, reward, done):
        self.store(next_obs, reward, done)
        return self.train_step()

    def _train_step(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: action selection by online net, evaluation by target net.
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + self.gamma * next_q * (1 - dones)

        # Huber loss is more robust to reward outliers than MSE.
        loss = nn.SmoothL1Loss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        return float(loss.item())

    def end_episode(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.reset_episode()

    def save(self, path=None, filename="weights.pth"):
        save_path = path or self.config.weights_dir
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.q_net.state_dict(), os.path.join(save_path, filename))

    def load(self) -> None:
        weights_path = os.path.join(self.config.weights_dir, "weights.pth")
        if not os.path.exists(weights_path):
            print(f"No weights found at {weights_path} -- starting fresh.")
            return
        self._pending_load_path = weights_path
        print(f"Weights queued for loading from {weights_path}")