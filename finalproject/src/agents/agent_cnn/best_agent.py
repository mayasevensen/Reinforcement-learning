"""
CNN-based Dueling Double DQN agent for the Collector environment.

Speed + quality changes vs the previous CNN agent:

  Architecture
    - Strided convs shrink spatial dims 16 -> 16 -> 8 -> 4 instead of
      keeping 16x16 throughout. Final flatten is 64*4*4 = 1024 features
      instead of 16384, which removes ~3.4M parameters from the
      first dense layer.
    - Total params now ~150K (was ~4.2M). Forward+backward is
      roughly an order of magnitude faster on CPU and several times
      faster on GPU.
    - Channels are now cleanly separated:
        Ch 0: obstacles only (binary)
        Ch 1: my position    (one-hot)
        Ch 2: opponent pos   (one-hot)
        Ch 3: items          (binary)
      Empty tiles are implied by all-zero. The previous design put
      tile_type/2 in ch 0, which mixed obstacles and items together.

  Replay buffer
    - Pre-allocated numpy arrays. No per-step Python allocations,
      no np.stack at sample time. Sampling is just fancy indexing.
    - Reduces wall-clock per train_step noticeably, especially as
      the buffer fills up.

  Target network
    - Polyak (soft) updates with tau=0.005 every train step instead
      of a hard copy every 500 steps. Smoother targets, no jumps.

  Exploration
    - Epsilon decays per *training step*, not per episode. Decoupled
      from episode length, which varies from 200 to 800 here.

  Everything else preserved
    - Double DQN target computation
    - Dueling value/advantage heads
    - Huber (SmoothL1) loss
    - Gradient clipping at 1.0
    - Same epsilon-greedy action selection
    - Same load() / save() / act() / store() / train_step() interface
      so trainCNN.py barely changes.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from types import SimpleNamespace

from agents.agent_base import BaseAgent
from environments.collector.state import EnvState

# import warnings
# warnings.filterwarnings("ignore")
# os.environ["PYTHONWARNINGS"] = "ignore"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRID_H = 16
GRID_W = 16
GLOBAL_DIM = 10  # my_score, opp_score, items_on_map, steps, opp_dist, opp_closer,
                 # + 4-dim one-hot of BFS-suggested next action toward nearest item
N_ACTIONS = 4


# ---------------------------------------------------------------------------
# BFS navigation helper
# ---------------------------------------------------------------------------
def bfs_next_action(tile_map, start):
    """
    BFS from `start` to the nearest reachable item on `tile_map`.
    Returns a one-hot np.float32 array of shape (4,) where the hot index
    is the first step on the shortest path:
        0=up, 1=right, 2=down, 3=left
    Returns all-zeros if there are no reachable items or agent is already
    on one (shouldn't happen, but safe).

    Action deltas match the environment convention:
        0: up    -> dy=-1, dx= 0
        1: right -> dy= 0, dx=+1
        2: down  -> dy=+1, dx= 0
        3: left  -> dy= 0, dx=-1
    """
    H, W = tile_map.shape
    sy, sx = int(start[0]), int(start[1])
    one_hot = np.zeros(4, dtype=np.float32)

    if not (0 <= sy < H and 0 <= sx < W) or tile_map[sy, sx] == 1:
        return one_hot

    # BFS: state = (y, x), track parent to reconstruct first step
    from collections import deque
    parent = {(sy, sx): None}   # (y,x) -> (parent_y, parent_x) or None for start
    first_action = {}            # (y,x) -> action taken from start to reach it
    queue = deque([(sy, sx)])

    deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left

    while queue:
        y, x = queue.popleft()

        if tile_map[y, x] == 2 and (y, x) != (sy, sx):
            # Found nearest item — reconstruct first action
            # Walk back to find the node whose parent is the start
            cur = (y, x)
            while parent[cur] != (sy, sx):
                cur = parent[cur]
            one_hot[first_action[cur]] = 1.0
            return one_hot

        for a, (dy, dx) in enumerate(deltas):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and (ny, nx) not in parent \
                    and tile_map[ny, nx] != 1:
                parent[(ny, nx)] = (y, x)
                first_action[(ny, nx)] = first_action.get((y, x), a) \
                    if (y, x) != (sy, sx) else a
                queue.append((ny, nx))

    return one_hot  # no reachable item

# Frame stacking: how many recent frames of (me, opp) positions to keep.
# Static channels (obstacles, items) are not stacked -- they barely change
# and the global feature `items_on_map` already tracks item depletion.
#
# Stacked channel layout:
#   ch 0          : obstacles (binary)
#   ch 1          : items (binary)
#   ch 2..2+K-1   : my position over last K frames (newest first)
#   ch 2+K..end   : opponent position over last K frames (newest first)
FRAME_STACK = 2
N_CHANNELS  = 2 + 2 * FRAME_STACK   # = 6 with FRAME_STACK=2


# ---------------------------------------------------------------------------
# Observation preprocessing
# ---------------------------------------------------------------------------
def _raw_features(obs):
    """
    Extract the four single-frame channels and the global feature vector
    from a raw env observation. This is the building block the FrameStacker
    composes into the final stacked observation.

    Returns:
      ch_obs   : (H, W) float32  -- obstacle mask, static-ish
      ch_items : (H, W) float32  -- item mask, decreases over time
      ch_me    : (H, W) float32  -- my position one-hot
      ch_opp   : (H, W) float32  -- opponent position one-hot
      global_vec : (GLOBAL_DIM,) float32
    """
    raw_map = obs['map_features']['tile_type']
    H, W    = raw_map.shape
    my_pos  = obs['units']['position'][0]
    opp_pos = obs['units']['position'][1]

    ch_obs   = (raw_map == 1).astype(np.float32)
    ch_items = (raw_map == 2).astype(np.float32)

    ch_me = np.zeros((H, W), dtype=np.float32)
    my_y, my_x = int(my_pos[0]), int(my_pos[1])
    if 0 <= my_y < H and 0 <= my_x < W:
        ch_me[my_y, my_x] = 1.0

    ch_opp = np.zeros((H, W), dtype=np.float32)
    op_y, op_x = int(opp_pos[0]), int(opp_pos[1])
    if 0 <= op_y < H and 0 <= op_x < W:
        ch_opp[op_y, op_x] = 1.0

    diag = float(H + W)
    team_pts   = obs['team_points'].astype(np.float32).flatten()
    my_score   = float(team_pts[0]) / 50.0
    opp_score  = float(team_pts[1]) / 50.0
    items_norm = float(obs['items_on_map'].item()) / 50.0
    steps_norm = float(obs['steps'].item()) / 1000.0
    opp_dist   = float(np.abs(opp_pos - my_pos).sum()) / diag

    item_locs = np.argwhere(raw_map == 2)
    if len(item_locs) > 0:
        my_d  = float(np.abs(item_locs - my_pos).sum(axis=1).min())
        opp_d = float(np.abs(item_locs - opp_pos).sum(axis=1).min())
        opp_closer = 1.0 if opp_d < my_d else 0.0
    else:
        opp_closer = 0.0

    # BFS-suggested next action toward nearest item (4-dim one-hot)
    bfs_action = bfs_next_action(raw_map, my_pos)

    global_vec = np.concatenate([
        np.array([my_score, opp_score, items_norm, steps_norm, opp_dist, opp_closer],
                 dtype=np.float32),
        bfs_action,   # 4 dims: one-hot of recommended action
    ])                # total: GLOBAL_DIM = 10

    return ch_obs, ch_items, ch_me, ch_opp, global_vec


class FrameStacker:
    """
    Maintains a rolling history of (me, opp) position channels and assembles
    the stacked observation:
        [obstacles, items, me_t, me_{t-1}, ..., me_{t-K+1},
                          opp_t, opp_{t-1}, ..., opp_{t-K+1}]

    On reset(), the first frame is repeated K times to fill the history,
    so the agent always has a valid K-frame stack from the very first step.

    One stacker per "view" -- the agent uses one for its own observations,
    and FrozenSelfOpponent (in trainCNN.py) creates its own for self-play.
    """

    def __init__(self, k=FRAME_STACK):
        self.k = k
        self._me_hist  = None  # list of (H, W) arrays, newest first
        self._opp_hist = None

    def reset(self):
        self._me_hist  = None
        self._opp_hist = None

    def step(self, obs):
        """
        Process one observation. Returns a dict with keys "grid" and "global".
        Call this on every step (including the first); the stacker handles
        the cold-start case by repeating the first frame.
        """
        ch_obs, ch_items, ch_me, ch_opp, global_vec = _raw_features(obs)

        if self._me_hist is None:
            # Cold start: pad history with copies of the first frame.
            self._me_hist  = [ch_me]  * self.k
            self._opp_hist = [ch_opp] * self.k
        else:
            self._me_hist.insert(0,  ch_me)
            self._opp_hist.insert(0, ch_opp)
            if len(self._me_hist)  > self.k: self._me_hist.pop()
            if len(self._opp_hist) > self.k: self._opp_hist.pop()

        grid = np.stack(
            [ch_obs, ch_items] + self._me_hist + self._opp_hist,
            axis=0
        ).astype(np.float32, copy=False)

        return {"grid": grid, "global": global_vec}

    def peek(self, obs):
        """
        Compute what step(obs) WOULD return without mutating the history.
        Used to compute next_state for the replay buffer in agent.store(),
        so the stacker's state stays correct for the next call to step().
        """
        ch_obs, ch_items, ch_me, ch_opp, global_vec = _raw_features(obs)

        if self._me_hist is None:
            me_hist  = [ch_me]  * self.k
            opp_hist = [ch_opp] * self.k
        else:
            me_hist  = [ch_me]  + self._me_hist[:self.k - 1]
            opp_hist = [ch_opp] + self._opp_hist[:self.k - 1]

        grid = np.stack(
            [ch_obs, ch_items] + me_hist + opp_hist,
            axis=0
        ).astype(np.float32, copy=False)

        return {"grid": grid, "global": global_vec}


def preprocess_obs(obs):
    """
    Stateless single-frame preprocessing. Used by self-play opponents that
    don't carry their own stacker (each FrozenSelfOpponent should use a
    FrameStacker; see trainCNN.py). Kept here for backwards compatibility
    with any code that calls it directly.

    Returns the same K-frame layout as FrameStacker.step(), but with the
    history channels filled by repeating the current frame.
    """
    ch_obs, ch_items, ch_me, ch_opp, global_vec = _raw_features(obs)
    grid = np.stack(
        [ch_obs, ch_items] + [ch_me] * FRAME_STACK + [ch_opp] * FRAME_STACK,
        axis=0
    ).astype(np.float32, copy=False)
    return {"grid": grid, "global": global_vec}


# ---------------------------------------------------------------------------
# Network: small strided CNN + dueling heads
# ---------------------------------------------------------------------------
class CNNQNetwork(nn.Module):
    """
    Strided conv backbone:
      Conv(N_CHANNELS -> 32, 3x3, stride=1, pad=1)  -> 16x16
      Conv(32 -> 64,         3x3, stride=2, pad=1)  ->  8x8
      Conv(64 -> 64,         3x3, stride=2, pad=1)  ->  4x4
      Flatten -> 64*4*4 = 1024

    Then merge with the global feature vector and feed dueling heads.

    With FRAME_STACK=2 we have N_CHANNELS=6 inputs. Total ~220K params.
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

        cnn_out_dim = 64 * 4 * 4   # 1024 for a 16x16 input

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
        grid       : (B, 4, H, W)
        global_vec : (B, GLOBAL_DIM)
        """
        h = self.cnn(grid)
        h = h.flatten(start_dim=1)
        h = torch.cat([h, global_vec], dim=1)
        h = self.merger(h)

        v = self.value_head(h)
        a = self.advantage_head(h)
        return v + (a - a.mean(dim=1, keepdim=True))


# ---------------------------------------------------------------------------
# Pre-allocated replay buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """
    Pre-allocated numpy arrays for fast sampling.

    Memory usage at capacity=50000, 10x16x16 grid, float32:
      grids:      50000 * 10 * 16 * 16 * 4 bytes ~= 195 MB  (N_CHANNELS=10)
      next_grids: same                           ~= 195 MB
      globals:    50000 * 10 * 4                 ~= 2.0 MB   (GLOBAL_DIM=10)
      etc.
    Total ~400 MB, which is fine for a training run. If RAM is tight
    you can drop capacity to e.g. 20000.
    """

    def __init__(self, capacity, grid_shape, global_dim):
        self.capacity = capacity
        self.size = 0
        self.idx = 0

        C, H, W = grid_shape
        self.grids       = np.zeros((capacity, C, H, W), dtype=np.float32)
        self.globals_    = np.zeros((capacity, global_dim), dtype=np.float32)
        self.actions     = np.zeros((capacity,), dtype=np.int64)
        self.rewards     = np.zeros((capacity,), dtype=np.float32)
        self.next_grids  = np.zeros((capacity, C, H, W), dtype=np.float32)
        self.next_globs  = np.zeros((capacity, global_dim), dtype=np.float32)
        self.dones       = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        i = self.idx
        self.grids[i]      = state["grid"]
        self.globals_[i]   = state["global"]
        self.actions[i]    = action
        self.rewards[i]    = reward
        self.next_grids[i] = next_state["grid"]
        self.next_globs[i] = next_state["global"]
        self.dones[i]      = float(done)

        self.idx  = (self.idx + 1) % self.capacity
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

        # Exploration
        self.epsilon       = getattr(config, 'epsilon_start',     1.0)
        self.epsilon_end   = getattr(config, 'epsilon_end',       0.05)
        # Per-step decay: roughly hits epsilon_end after ~epsilon_decay_steps env steps.
        self.epsilon_decay_steps = getattr(config, 'epsilon_decay_steps', 400_000)

        # Optimisation
        self.gamma        = getattr(config, 'gamma',          0.99)
        self.batch_size   = getattr(config, 'batch_size',     128)
        self.lr           = getattr(config, 'learning_rate',  3e-4)
        self.tau          = getattr(config, 'tau',            0.005)  # soft target update
        self.training     = getattr(config, 'training',       False)
        self.hidden_dim   = getattr(config, 'hidden_dim',     128)

        # Buffer
        self.buffer_size     = getattr(config, 'buffer_size',     50000)
        self.min_buffer_size = getattr(config, 'min_buffer_size', 5000)

        # Networks built lazily on first act() so we can load weights cleanly
        self.q_net      = None
        self.target_net = None
        self.optimizer  = None
        self.replay_buffer = None  # built when training starts

        self._step_count   = 0     # env steps stored
        self._train_steps  = 0     # gradient updates done
        self._last_state   = None
        self._last_action  = None
        self._epsilon_start_value = self.epsilon  # for per-step linear decay

        # Frame stacker maintains the rolling history of (me, opp) channels.
        # Reset between episodes so history doesn't leak across resets.
        self._stacker = FrameStacker(k=FRAME_STACK)

    # -- Network setup --------------------------------------------------------
    def _build_networks(self):
        self.q_net      = CNNQNetwork(GLOBAL_DIM, self.hidden_dim).to(self.device)
        self.target_net = CNNQNetwork(GLOBAL_DIM, self.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer  = optim.Adam(self.q_net.parameters(), lr=self.lr)

        if self.training and self.replay_buffer is None:
            self.replay_buffer = ReplayBuffer(
                capacity=self.buffer_size,
                grid_shape=(N_CHANNELS, GRID_H, GRID_W),
                global_dim=GLOBAL_DIM,
            )

    def reset_episode(self):
        self._last_state  = None
        self._last_action = None
        self._stacker.reset()

    def _state_to_tensors(self, state):
        grid = torch.from_numpy(state["grid"]).unsqueeze(0).to(self.device)
        gvec = torch.from_numpy(state["global"]).unsqueeze(0).to(self.device)
        return grid, gvec

    # -- Action selection -----------------------------------------------------
    def act(self, observation: EnvState) -> int:
        # Advance the frame stacker. From here on, self._stacker holds
        # the history including this observation as the newest frame.
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
            action = random.randint(0, N_ACTIONS - 1)
        else:
            with torch.no_grad():
                grid, gvec = self._state_to_tensors(state)
                action = int(self.q_net(grid, gvec).argmax(dim=1).item())

        self._last_state  = state
        self._last_action = action
        return action

    # -- Storing transitions --------------------------------------------------
    def store(self, next_obs, reward, done):
        if not self.training or self._last_state is None:
            return
        # Use peek() to compute what the stacked next_state would look like
        # without mutating the stacker -- the next act() call will properly
        # advance it via step().
        next_state = self._stacker.peek(next_obs)
        self.replay_buffer.push(
            self._last_state, self._last_action, reward, next_state, float(done)
        )
        self._step_count += 1
        # Per-step linear epsilon decay
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
        self.store(next_obs, reward, done)
        return self.train_step()

    def _train_step(self):
        grids, globs, actions, rewards, n_grids, n_globs, dones = \
            self.replay_buffer.sample(self.batch_size)

        grids   = torch.from_numpy(grids).to(self.device)
        globs   = torch.from_numpy(globs).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        n_grids = torch.from_numpy(n_grids).to(self.device)
        n_globs = torch.from_numpy(n_globs).to(self.device)
        dones   = torch.from_numpy(dones).to(self.device)

        q_values = self.q_net(grids, globs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target
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

        # Soft target update (Polyak)
        with torch.no_grad():
            for p, tp in zip(self.q_net.parameters(), self.target_net.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

        self._train_steps += 1
        return float(loss.item())

    def end_episode(self):
        # Epsilon already decays per step now; keep the hook for compatibility.
        self.reset_episode()

    # -- I/O ------------------------------------------------------------------
    def save(self, path=None, filename="weights.pth"):
        save_path = path or self.config.weights_dir
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.q_net.state_dict(), os.path.join(save_path, filename))

    def load(self) -> None:
        weights_path = os.path.join(self.config.weights_dir, "weights.pth")
        if not os.path.exists(weights_path):
            print(f"No CNN weights found at {weights_path} -- starting fresh.")
            return
        self._pending_load_path = weights_path
        print(f"CNN weights queued for loading from {weights_path}")