"""
CNN-based Dueling Double DQN agent for the Collector environment.

Architecture
  - Strided CNN backbone (16x16 -> 8x8 -> 4x4) feeding dueling
    value/advantage heads.
  - Input is a 6-frame stack of 4 spatial channels
    (obstacles, items, my position, opponent position) plus a 10-dim
    vector of normalised scalars including a BFS-suggested next action
    toward the nearest reachable item.

Training
  - Double DQN target with Polyak (soft) updates of the target net.
  - Huber (SmoothL1) loss with gradient clipping at 1.0.
  - Pre-allocated numpy replay buffer for fast sampling.
  - Epsilon-greedy exploration with per-step linear decay.

Public interface
  - act(obs) / store(next_obs, reward, done) / train_step()
  - reset_episode() / end_episode()
  - save() / load()
"""

import os
import random
from collections import deque
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.agent_base import BaseAgent
from environments.collector.state import EnvState


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRID_H = 16
GRID_W = 16
N_ACTIONS = 4

# Global feature vector layout (10 dims total):
#   [my_score, opp_score, items_on_map, steps, opp_dist, opp_closer (with manhatten)]
#   + 4-dim one-hot of BFS-suggested next action toward nearest item
GLOBAL_DIM = 10

# Frame stacking: how many recent frames of (me, opp) positions to keep.
# Static channels (obstacles, items) are not stacked (no change)
# and the global feature `items_on_map` already tracks item depletion.
#
# Stacked grid layout (N_CHANNELS = 2 + 2*FRAME_STACK):
#   ch 0          : obstacles (binary)
#   ch 1          : items (binary)
#   ch 2..2+K-1   : my position over last K frames (newest first)
#   ch 2+K..end   : opponent position over last K frames (newest first)
FRAME_STACK = 3
N_CHANNELS = 2 + 2 * FRAME_STACK


# BFS navigation helper
# Action conventions used throughout this file:
# [up, right, down, left]
_BFS_DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

BFS_EXPLORE_PROB = 0.4  # share of random actions that follow the BFS hint


def bfs_next_action(tile_map, start):
    """
    BFS from `start` to the nearest reachable item on `tile_map`.
    Returns a one-hot array of length 4 indicating the first step
    on the shortest path (up/right/down/left). Returns all-zeros if no
    item is reachable or `start` is invalid.

    Treats tile_type == 1 as an obstacle and tile_type == 2 as an item.
    """
    H, W = tile_map.shape
    sy, sx = int(start[0]), int(start[1])
    one_hot = np.zeros(4, dtype=np.float32)

    if not (0 <= sy < H and 0 <= sx < W) or tile_map[sy, sx] == 1:
        return one_hot

    parent = {(sy, sx): None}  # for reconstructing the path
    first_action = {}  # (y, x) -> first action taken from start
    queue = deque([(sy, sx)])

    while queue:
        y, x = queue.popleft()

        # Goal test: an item that isn't the start cell.
        if tile_map[y, x] == 2 and (y, x) != (sy, sx):
            cur = (y, x)
            while parent[cur] != (sy, sx):
                cur = parent[cur]
            one_hot[first_action[cur]] = 1.0
            return one_hot

        for a, (dy, dx) in enumerate(_BFS_DELTAS):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and (ny, nx) not in parent \
                    and tile_map[ny, nx] != 1:
                parent[(ny, nx)] = (y, x)
                # Inherit the first-action label from the parent, except
                # at the start where the first action *is* `a`.
                first_action[(ny, nx)] = a if (y, x) == (sy, sx) \
                    else first_action[(y, x)]
                queue.append((ny, nx))

    return one_hot  # no reachable item


# ---------------------------------------------------------------------------
# Observation preprocessing
# ---------------------------------------------------------------------------
def _raw_features(obs):
    """
    Extract the four single-frame spatial channels and the 10-dim global
    feature vector from a raw env observation.

    Returns:
      ch_obs: (H, W) float32  - obstacle mask
      ch_items: (H, W) float32  - item mask
      ch_me: (H, W) float32  - my position one-hot
      ch_opp: (H, W) float32  - opponent position one-hot
      global_vec: (GLOBAL_DIM,) float32
    """
    raw_map = obs['map_features']['tile_type']
    H, W = raw_map.shape
    my_pos = obs['units']['position'][0]
    opp_pos = obs['units']['position'][1]

    ch_obs = (raw_map == 1).astype(np.float32)
    ch_items = (raw_map == 2).astype(np.float32)

    ch_me = np.zeros((H, W), dtype=np.float32)
    my_y, my_x = int(my_pos[0]), int(my_pos[1])
    if 0 <= my_y < H and 0 <= my_x < W:
        ch_me[my_y, my_x] = 1.0

    ch_opp = np.zeros((H, W), dtype=np.float32)
    op_y, op_x = int(opp_pos[0]), int(opp_pos[1])
    if 0 <= op_y < H and 0 <= op_x < W:
        ch_opp[op_y, op_x] = 1.0

    # Global features, all normalised so they live in roughly [0, 1].
    diag = float(H + W)
    team_pts = obs['team_points'].astype(np.float32).flatten()
    my_score = float(team_pts[0]) / 50.0
    opp_score = float(team_pts[1]) / 50.0
    items_norm = float(obs['items_on_map'].item()) / 50.0
    steps_norm = float(obs['steps'].item()) / 1000.0
    opp_dist = float(np.abs(opp_pos - my_pos).sum()) / diag

    item_locs = np.argwhere(raw_map == 2)
    if len(item_locs) > 0:
        my_d = float(np.abs(item_locs - my_pos).sum(axis=1).min())
        opp_d = float(np.abs(item_locs - opp_pos).sum(axis=1).min())
        opp_closer = 1.0 if opp_d < my_d else 0.0
    else:
        opp_closer = 0.0

    bfs_action = bfs_next_action(raw_map, my_pos)

    global_vec = np.concatenate([
        np.array([my_score, opp_score, items_norm, steps_norm,
                  opp_dist, opp_closer], dtype=np.float32),
        bfs_action,
    ])
    return ch_obs, ch_items, ch_me, ch_opp, global_vec


class FrameStacker:
    """
    Maintains a rolling K-frame history of (me, opp) position channels
    and assembles the stacked observation:
        [obstacles, items,
         me_t, me_{t-1}, ..., me_{t-K+1},
         opp_t, opp_{t-1}, ..., opp_{t-K+1}]

    Cold start is handled by repeating the first frame K times, so the
    network always sees a valid K-frame stack from the very first step.

    Each "view" needs its own stacker — the agent uses one for itself,
    and self-play opponents create their own.
    """

    def __init__(self, k=FRAME_STACK):
        self.k = k
        self._me_hist = None  # list of (H, W) arrays, newest first
        self._opp_hist = None

    def reset(self):
        # Called between episodes so history doesn't leak across resets.
        self._me_hist = None
        self._opp_hist = None

    def step(self, obs):
        """Process one observation and advance the history. Returns {grid, global}."""
        ch_obs, ch_items, ch_me, ch_opp, global_vec = _raw_features(obs)

        if self._me_hist is None:
            # Cold start: pad history with copies of the first frame.
            self._me_hist = [ch_me] * self.k
            self._opp_hist = [ch_opp] * self.k
        else:
            self._me_hist.insert(0, ch_me)
            self._opp_hist.insert(0, ch_opp)
            if len(self._me_hist) > self.k: self._me_hist.pop()
            if len(self._opp_hist) > self.k: self._opp_hist.pop()

        grid = np.stack(
            [ch_obs, ch_items] + self._me_hist + self._opp_hist,
            axis=0,
        ).astype(np.float32, copy=False)
        return {"grid": grid, "global": global_vec}

    def peek(self, obs):
        """
        Same return value as step() but without mutating the history.
        Used inside store() so the next call to act() sees the right history.
        """
        ch_obs, ch_items, ch_me, ch_opp, global_vec = _raw_features(obs)

        if self._me_hist is None:
            me_hist = [ch_me] * self.k
            opp_hist = [ch_opp] * self.k
        else:
            me_hist = [ch_me] + self._me_hist[:self.k - 1]
            opp_hist = [ch_opp] + self._opp_hist[:self.k - 1]

        grid = np.stack(
            [ch_obs, ch_items] + me_hist + opp_hist,
            axis=0,
        ).astype(np.float32, copy=False)
        return {"grid": grid, "global": global_vec}


def preprocess_obs(obs):
    """
    Stateless single-frame preprocessing: a stacked observation in which
    the K position-history slots are filled by repeating the current frame.
    Useful for callers that don't want to maintain their own FrameStacker.
    """
    ch_obs, ch_items, ch_me, ch_opp, global_vec = _raw_features(obs)
    grid = np.stack(
        [ch_obs, ch_items] + [ch_me] * FRAME_STACK + [ch_opp] * FRAME_STACK,
        axis=0,
    ).astype(np.float32, copy=False)
    return {"grid": grid, "global": global_vec}


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class CNNQNetwork(nn.Module):
    """
    Strided CNN backbone + dueling value/advantage heads.

    Backbone (input is N_CHANNELS x 16 x 16):
        Conv 3x3 stride 1, pad 1: N_CHANNELS -> 32  (16 x 16)
        Conv 3x3 stride 2, pad 1: 32  -> 64  (8 x 8)
        Conv 3x3 stride 2, pad 1: 64  -> 64  (4 x 4)
    The final 64*4*4 = 1024-dim feature is concatenated with the global
    vector and fed to two MLPs that produce V(s) and A(s, ·). Q(s, a)
    is reassembled as V + (A - mean(A)).
    """

    def __init__(self, global_dim, hidden_dim=128, output_dim=N_ACTIONS):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(N_CHANNELS, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        cnn_out_dim = 64 * 4 * 4  # for a 16x16 input

        self.merger = nn.Sequential(
            nn.Linear(cnn_out_dim + global_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, grid, global_vec):
        """
        grid: (B, N_CHANNELS, H, W)
        global_vec: (B, GLOBAL_DIM)
        returns: (B, N_ACTIONS) Q-values
        """
        h = self.cnn(grid).flatten(start_dim=1)
        h = torch.cat([h, global_vec], dim=1)
        h = self.merger(h)

        v = self.value_head(h)
        a = self.advantage_head(h)
        # Subtracting mean(A) removes the V/A gauge ambiguity and stabilises training.
        return v + (a - a.mean(dim=1, keepdim=True))


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """
    Pre-allocated circular buffer. Storing into fixed numpy arrays avoids
    per-step Python allocations, and sampling is just fancy indexing.
    """

    def __init__(self, capacity, grid_shape, global_dim):
        self.capacity = capacity
        self.size = 0
        self.idx = 0

        C, H, W = grid_shape
        self.grids = np.zeros((capacity, C, H, W), dtype=np.float32)
        self.globals_ = np.zeros((capacity, global_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_grids = np.zeros((capacity, C, H, W), dtype=np.float32)
        self.next_globs = np.zeros((capacity, global_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        i = self.idx
        self.grids[i] = state["grid"]
        self.globals_[i] = state["global"]
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_grids[i] = next_state["grid"]
        self.next_globs[i] = next_state["global"]
        self.dones[i] = float(done)

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.grids[idxs],
            self.globals_[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_grids[idxs],
            self.next_globs[idxs],
            self.dones[idxs],
        )

    def __len__(self):
        return self.size


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class Agent(BaseAgent):
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Exploration: epsilon decays linearly from start to end over
        # epsilon_decay_steps env steps (driven from store()).
        self.epsilon = getattr(config, 'epsilon_start', 1.0)
        self.epsilon_end = getattr(config, 'epsilon_end', 0.05)
        self.epsilon_decay_steps = getattr(config, 'epsilon_decay_steps', 400_000)
        self._epsilon_start_value = self.epsilon

        # Optimisation
        self.gamma = getattr(config, 'gamma', 0.99)
        self.batch_size = getattr(config, 'batch_size', 128)
        self.lr = getattr(config, 'learning_rate', 3e-4)
        self.tau = getattr(config, 'tau', 0.005)  # Polyak coefficient
        self.training = getattr(config, 'training', False)
        self.hidden_dim = getattr(config, 'hidden_dim', 128)

        # Replay buffer
        self.buffer_size = getattr(config, 'buffer_size', 50000)
        self.min_buffer_size = getattr(config, 'min_buffer_size', 5000)

        # Networks are built lazily on first act() so load() can queue a
        # weights file before the network exists.
        self.q_net = None
        self.target_net = None
        self.optimizer = None
        self.replay_buffer = None

        self._step_count = 0  # env steps stored
        self._train_steps = 0  # gradient updates done
        self._last_state = None
        self._last_action = None

        self._stacker = FrameStacker(k=FRAME_STACK)

    # -- Network setup --------------------------------------------------------
    def _build_networks(self):
        self.q_net = CNNQNetwork(GLOBAL_DIM, self.hidden_dim).to(self.device)
        self.target_net = CNNQNetwork(GLOBAL_DIM, self.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)

        if self.training and self.replay_buffer is None:
            self.replay_buffer = ReplayBuffer(
                capacity=self.buffer_size,
                grid_shape=(N_CHANNELS, GRID_H, GRID_W),
                global_dim=GLOBAL_DIM,
            )

    def reset_episode(self):
        self._last_state = None
        self._last_action = None
        self._stacker.reset()

    def end_episode(self):
        # Epsilon decays per env step (in store()), so this is just a hook.
        self.reset_episode()

    def _state_to_tensors(self, state):
        grid = torch.from_numpy(state["grid"]).unsqueeze(0).to(self.device)
        gvec = torch.from_numpy(state["global"]).unsqueeze(0).to(self.device)
        return grid, gvec

    # -- Action selection -----------------------------------------------------
    def act(self, observation: EnvState) -> int:
        state = self._stacker.step(observation)

        if self.q_net is None:
            self._build_networks()
            if hasattr(self, '_pending_load_path'):
                try:
                    sd = torch.load(self._pending_load_path, map_location=self.device)
                    self.q_net.load_state_dict(sd)
                    self.target_net.load_state_dict(self.q_net.state_dict())
                    if not self.training:
                        self.q_net.eval()
                    print("CNN weights loaded!")
                except Exception as e:
                    print(f"Could not load CNN weights ({e}); starting fresh.")
                del self._pending_load_path

        if self.training and random.random() < self.epsilon:
            # Exploration: BFS-biased random action.
            # The last 4 entries of `global` are a one-hot of the BFS-suggested
            # next step toward the nearest reachable item. We follow it with
            # probability BFS_EXPLORE_PROB; otherwise pick uniformly. If BFS
            # has no suggestion (all-zero one-hot), fall back to uniform.
            bfs_one_hot = state["global"][-4:]
            if random.random() < BFS_EXPLORE_PROB and bfs_one_hot.sum() > 0:
                action = int(np.argmax(bfs_one_hot))
            else:
                action = random.randint(0, N_ACTIONS - 1)
        else:
            with torch.no_grad():
                grid, gvec = self._state_to_tensors(state)
                action = int(self.q_net(grid, gvec).argmax(dim=1).item())

        self._last_state = state
        self._last_action = action
        return action

    # -- Storing transitions --------------------------------------------------
    def store(self, next_obs, reward, done):
        if not self.training or self._last_state is None:
            return

        # peek() so the stacker isn't double-advanced before the next act().
        next_state = self._stacker.peek(next_obs)
        self.replay_buffer.push(
            self._last_state, self._last_action, reward, next_state, float(done)
        )
        self._step_count += 1

        # Linear epsilon decay tied to env steps, not episodes (episode
        # lengths are randomised in training, which would otherwise make
        # decay rate non-uniform).
        frac = min(1.0, self._step_count / float(self.epsilon_decay_steps))
        self.epsilon = self._epsilon_start_value + frac * (
            self.epsilon_end - self._epsilon_start_value
        )

    # -- Training -------------------------------------------------------------
    def train_step(self):
        if not self.training or len(self.replay_buffer) < self.min_buffer_size:
            return None
        return self._train_step()

    def update(self, next_obs, reward, done):
        """Convenience: store() followed by train_step()."""
        self.store(next_obs, reward, done)
        return self.train_step()

    def _train_step(self):
        grids, globs, actions, rewards, n_grids, n_globs, dones = \
            self.replay_buffer.sample(self.batch_size)

        grids = torch.from_numpy(grids).to(self.device)
        globs = torch.from_numpy(globs).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        n_grids = torch.from_numpy(n_grids).to(self.device)
        n_globs = torch.from_numpy(n_globs).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)

        q_values = self.q_net(grids, globs).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        # Double DQN: live net picks the best next action, target net evaluates it.
        with torch.no_grad():
            next_actions = self.q_net(n_grids, n_globs).argmax(dim=1)
            next_q = self.target_net(n_grids, n_globs).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            targets = rewards + self.gamma * next_q * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # Polyak (soft) target update: target <- (1 - tau) * target + tau * live.
        with torch.no_grad():
            for p, tp in zip(self.q_net.parameters(),
                             self.target_net.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

        self._train_steps += 1
        return float(loss.item())

    # -- I/O ------------------------------------------------------------------
    def save(self, path=None, filename="weights.pth"):
        save_path = path or self.config.weights_dir
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.q_net.state_dict(), os.path.join(save_path, filename))

    def load(self) -> None:
        # Queue the weights file; act() does the actual load once the
        # network has been built. This lets the same Agent class work
        # both for fresh training (no file yet) and for evaluation.
        weights_path = os.path.join(self.config.weights_dir, "weights.pth")
        if not os.path.exists(weights_path):
            print(f"No CNN weights found at {weights_path} -- starting fresh.")
            return
        self._pending_load_path = weights_path
        print(f"CNN weights queued for loading from {weights_path}")