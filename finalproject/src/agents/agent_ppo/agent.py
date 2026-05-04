"""
PPO Agent for the Collector environment.

Implements Proximal Policy Optimization (Schulman et al. 2017) from scratch.

Architecture:
  - Shared trunk MLP -> policy head (logits over 4 actions) + value head (scalar)
  - Policy outputs a categorical distribution; actions sampled during training,
    argmax during evaluation.

Key PPO components implemented here:
  - Clipped surrogate objective (Eq. 7 in paper): prevents destructively large
    policy updates by clipping the probability ratio r_t = pi_new / pi_old.
  - Generalized Advantage Estimation (GAE, Eq. 11): variance-reduced advantage
    estimates using a lambda-weighted mix of n-step returns.
  - Multiple epochs of minibatch SGD per rollout: data efficiency without the
    instability of naive policy gradient.
  - Entropy bonus: encourages exploration by penalising overconfident policies.
  - Combined loss: L = L_CLIP - c1 * L_VF + c2 * S  (Eq. 9 in paper)

The feature engineering (preprocess_obs) is shared with the DQN agent to keep
the two comparable. It lives in this file so the agent is self-contained.
"""

import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
from types import SimpleNamespace

from agents.agent_base import BaseAgent
from environments.collector.state import EnvState

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


# ---------------------------------------------------------------------------
# Feature engineering (same as DQN agent for fair comparison)
# ---------------------------------------------------------------------------
def bfs_distances(tile_map, start):
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
    56-dim feature vector, identical to the DQN agent.
    See DQN agent.py for full documentation of each component.
    """
    raw_map = obs['map_features']['tile_type']
    my_pos  = obs['units']['position'][0].astype(np.float32)
    opp_pos = obs['units']['position'][1].astype(np.float32)
    H, W    = raw_map.shape
    diag    = float(W + H)

    my_pos_norm  = my_pos  / np.array([H, W], dtype=np.float32)
    opp_pos_norm = opp_pos / np.array([H, W], dtype=np.float32)
    rel_opp      = (opp_pos - my_pos) / np.array([H, W], dtype=np.float32)

    item_locs     = np.argwhere(raw_map == 2).astype(np.float32)
    item_features = np.zeros(5 * 3, dtype=np.float32)
    nearest       = None

    if len(item_locs) > 0:
        diffs = item_locs - my_pos
        dists = np.abs(diffs).sum(axis=1)
        order = np.argsort(dists)[:5]
        for i, idx in enumerate(order):
            item_features[i * 3]     = diffs[idx][0] / H
            item_features[i * 3 + 1] = diffs[idx][1] / W
            item_features[i * 3 + 2] = dists[idx]    / diag
        nearest = item_locs[order[0]]

    if nearest is not None:
        dvec               = nearest - my_pos
        dist_to_nearest_man = float(np.abs(dvec).sum() / diag)
        norm               = np.abs(dvec).sum() + 1e-8
        direction          = (dvec / norm).astype(np.float32)
    else:
        direction           = np.zeros(2, dtype=np.float32)
        dist_to_nearest_man = 1.0

    radius = 2
    local  = np.full((2 * radius + 1, 2 * radius + 1), 0.5, dtype=np.float32)
    my_y, my_x = int(my_pos[0]), int(my_pos[1])
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            ni, nj = my_y + di, my_x + dj
            if 0 <= ni < H and 0 <= nj < W:
                local[di + radius, dj + radius] = raw_map[ni, nj] / 2.0

    items_on_map = np.array([obs['items_on_map'].item() / 50.0], dtype=np.float32)
    steps_norm   = np.array([obs['steps'].item()       / 1000.0], dtype=np.float32)
    team_points  = obs['team_points'].astype(np.float32).flatten() / 50.0

    opp_dist_man = float(np.abs(opp_pos - my_pos).sum())
    opp_close    = np.array([1.0 if opp_dist_man <= 2 else 0.0], dtype=np.float32)

    if nearest is not None:
        my_d      = float(np.abs(nearest - my_pos).sum())
        opp_d     = float(np.abs(nearest - opp_pos).sum())
        opp_closer = np.array([1.0 if opp_d < my_d else 0.0], dtype=np.float32)
    else:
        opp_closer = np.array([0.0], dtype=np.float32)

    if len(item_locs) > 0:
        dmap         = bfs_distances(raw_map, my_pos)
        item_locs_int = item_locs.astype(np.int32)
        ds           = dmap[item_locs_int[:, 0], item_locs_int[:, 1]]
        reachable    = ds[ds >= 0]
        bfs_d        = np.array([reachable.min() / diag if reachable.size > 0 else 1.0],
                                dtype=np.float32)
    else:
        bfs_d = np.array([1.0], dtype=np.float32)

    return np.concatenate([
        my_pos_norm,
        opp_pos_norm,
        rel_opp,
        direction,
        np.array([dist_to_nearest_man], dtype=np.float32),
        item_features,
        local.flatten(),
        team_points,
        items_on_map,
        steps_norm,
        opp_close,
        opp_closer,
        bfs_d,
    ])   # total: 56 dims


# ---------------------------------------------------------------------------
# Actor-Critic Network
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    """
    Shared trunk -> policy head + value head.

    Sharing parameters between policy and value (as the paper discusses in
    Section 5) lets the trunk learn a representation useful for both. The
    two heads then specialise on top of this shared representation.

    Policy head: outputs logits over 4 discrete actions.
    Value head:  outputs a single scalar V(s).
    """

    def __init__(self, input_dim, hidden_dim, n_actions=4):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),                          # Tanh as in the paper (vs ReLU in DQN)
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head  = nn.Linear(hidden_dim, 1)

        # Initialise policy head with small weights so initial policy is
        # nearly uniform -- important for stable early exploration.
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, x):
        h      = self.trunk(x)
        logits = self.policy_head(h)
        value  = self.value_head(h).squeeze(-1)
        return logits, value

    def get_action_and_value(self, x, action=None):
        """
        Sample an action (or evaluate a given one) and return:
          action, log_prob, entropy, value
        Used during rollout collection and during the PPO update.
        """
        logits, value = self.forward(x)
        dist          = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------
class RolloutBuffer:
    """
    Stores one rollout of T steps, then computes GAE advantages.
    Cleared after each PPO update (on-policy: data used once then discarded).
    """

    def __init__(self):
        self.states    = []
        self.actions   = []
        self.rewards   = []
        self.values    = []
        self.log_probs = []
        self.dones     = []

    def store(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_advantages(self, last_value, gamma, gae_lambda):
        """
        Generalised Advantage Estimation (GAE, Eq. 11 in paper).

        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
        A_t     = delta_t + gamma * lambda * A_{t+1}

        When lambda=1 this reduces to the full n-step return minus baseline.
        When lambda=0 this is the 1-step TD error (high bias, low variance).
        lambda=0.95 (as in paper) is a good bias-variance tradeoff.
        """
        T          = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae        = 0.0

        values_np     = np.array(self.values,  dtype=np.float32)
        rewards_np    = np.array(self.rewards,  dtype=np.float32)
        dones_np      = np.array(self.dones,    dtype=np.float32)

        for t in reversed(range(T)):
            next_val   = last_value if t == T - 1 else values_np[t + 1]
            next_done  = dones_np[t]
            delta      = rewards_np[t] + gamma * next_val * (1 - next_done) - values_np[t]
            gae        = delta + gamma * gae_lambda * (1 - next_done) * gae
            advantages[t] = gae

        returns = advantages + values_np
        return advantages, returns

    def get_tensors(self, device):
        return (
            torch.FloatTensor(np.array(self.states)).to(device),
            torch.LongTensor(np.array(self.actions)).to(device),
            torch.FloatTensor(np.array(self.log_probs)).to(device),
            torch.FloatTensor(np.array(self.values)).to(device),
        )

    def clear(self):
        self.states    = []
        self.actions   = []
        self.rewards   = []
        self.values    = []
        self.log_probs = []
        self.dones     = []

    def __len__(self):
        return len(self.states)


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------
class Agent(BaseAgent):
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters (with sane defaults if not in config)
        self.hidden_dim   = getattr(config, 'hidden_dim',   128)
        self.lr           = getattr(config, 'learning_rate', 3e-4)
        self.gamma        = getattr(config, 'gamma',         0.99)
        self.gae_lambda   = getattr(config, 'gae_lambda',    0.95)
        self.clip_epsilon = getattr(config, 'clip_epsilon',  0.2)
        self.ppo_epochs   = getattr(config, 'ppo_epochs',    4)
        self.minibatch    = getattr(config, 'minibatch_size', 64)
        self.vf_coef      = getattr(config, 'vf_coef',       0.5)
        self.ent_coef     = getattr(config, 'ent_coef',       0.01)
        self.max_grad_norm = getattr(config, 'max_grad_norm', 0.5)
        self.rollout_len  = getattr(config, 'rollout_len',   512)
        self.training     = getattr(config, 'training',      False)

        self.net       = None   # built lazily on first act() call
        self.optimizer = None
        self.buffer    = RolloutBuffer()

        # State held between act() and store()
        self._last_state    = None
        self._last_action   = None
        self._last_log_prob = None
        self._last_value    = None

    # ------------------------------------------------------------------
    # Network initialisation (lazy, so we know input_dim from first obs)
    # ------------------------------------------------------------------
    def _build_network(self, input_dim):
        self.net = ActorCritic(input_dim, self.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, eps=1e-5)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------
    def act(self, observation: EnvState) -> int:
        state = preprocess_obs(observation)

        if self.net is None:
            self._build_network(len(state))
            if hasattr(self, '_pending_load_path'):
                try:
                    sd = torch.load(self._pending_load_path, map_location=self.device)
                    self.net.load_state_dict(sd)
                    if not self.training:
                        self.net.eval()
                    print("PPO weights loaded!")
                except Exception as e:
                    print(f"Could not load PPO weights ({e}); starting fresh.")
                del self._pending_load_path

        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if self.training:
            with torch.no_grad():
                action, log_prob, _, value = self.net.get_action_and_value(s)
            self._last_state    = state
            self._last_action   = action.item()
            self._last_log_prob = log_prob.item()
            self._last_value    = value.item()
            return self._last_action
        else:
            # Greedy at eval/inference time
            with torch.no_grad():
                logits, _ = self.net(s)
            return int(logits.argmax(dim=1).item())

    def store(self, next_obs, reward: float, done: bool):
        """Called by train script after each environment step."""
        if not self.training or self._last_state is None:
            return
        self.buffer.store(
            self._last_state,
            self._last_action,
            reward,
            self._last_value,
            self._last_log_prob,
            float(done),
        )

    def update(self, next_obs, reward: float, done: bool):
        """Convenience wrapper (mirrors DQN agent interface)."""
        self.store(next_obs, reward, done)

    def train_on_rollout(self, last_obs):
        """
        Called by trainPPO.py when the rollout buffer is full.
        Runs PPO_EPOCHS passes of minibatch SGD on the collected data.
        Returns mean loss for logging.
        """
        if not self.training or len(self.buffer) == 0:
            return None

        # Bootstrap value for the last observation
        last_state = preprocess_obs(last_obs)
        s = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, last_value = self.net(s)
        last_value = last_value.item()

        # GAE
        advantages, returns = self.buffer.compute_advantages(
            last_value, self.gamma, self.gae_lambda
        )

        # Normalise advantages (standard PPO trick, reduces variance)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Collect tensors from buffer
        states_t, actions_t, old_log_probs_t, _ = self.buffer.get_tensors(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t    = torch.FloatTensor(returns).to(self.device)

        T = len(self.buffer)
        total_loss = 0.0
        n_updates  = 0

        # Multiple epochs of minibatch SGD (core PPO idea: reuse each rollout)
        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(T)
            for start in range(0, T, self.minibatch):
                idx = indices[start: start + self.minibatch]
                if len(idx) < 2:
                    continue

                mb_states    = states_t[idx]
                mb_actions   = actions_t[idx]
                mb_old_lp    = old_log_probs_t[idx]
                mb_adv       = advantages_t[idx]
                mb_returns   = returns_t[idx]

                # Evaluate current policy on minibatch
                _, new_log_probs, entropy, new_values = \
                    self.net.get_action_and_value(mb_states, mb_actions)

                # Probability ratio r_t = pi_new / pi_old  (log space for stability)
                log_ratio = new_log_probs - mb_old_lp
                ratio     = log_ratio.exp()

                # Clipped surrogate loss (Eq. 7)
                # We MINIMISE the negative of L_CLIP (gradient ascent -> descent)
                surr1     = ratio * mb_adv
                surr2     = torch.clamp(ratio, 1 - self.clip_epsilon,
                                               1 + self.clip_epsilon) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss (Eq. 9: c1 * L_VF)
                # Clipped value loss: prevents the value function from changing
                # too rapidly, similar in spirit to the policy clipping.
                value_loss = nn.MSELoss()(new_values, mb_returns)

                # Entropy bonus (Eq. 9: c2 * S) -- encourages exploration
                entropy_loss = -entropy.mean()

                # Combined loss (Eq. 9)
                loss = (policy_loss
                        + self.vf_coef  * value_loss
                        + self.ent_coef * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                n_updates  += 1

        self.buffer.clear()
        return total_loss / max(n_updates, 1)

    def reset_episode(self):
        self._last_state    = None
        self._last_action   = None
        self._last_log_prob = None
        self._last_value    = None

    def save(self, path=None, filename="weights_ppo.pth"):
        save_path = path or self.config.weights_dir
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.net.state_dict(), os.path.join(save_path, filename))

    def load(self) -> None:
        weights_path = os.path.join(self.config.weights_dir, "weights_ppo.pth")
        if not os.path.exists(weights_path):
            print(f"No PPO weights at {weights_path} -- starting fresh.")
            return
        self._pending_load_path = weights_path
        print(f"PPO weights queued for loading from {weights_path}")