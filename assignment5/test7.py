"""
Exercise 5 - Q3: Deep Q-Learning for CookieDisasterEnv
=======================================================
Approximates q(s,a) with a neural network and solves the control
problem via Q-learning with experience replay and a target network.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# 1. FEATURE ENGINEERING
# ──────────────────────────────────────────────

def extract_features(obs: dict) -> np.ndarray:
    x = obs["agent"]["pos"]
    v = obs["agent"]["vel"]
    c = obs["cookie"]["pos"]
    t = obs["cookie"]["time"]

    delta     = c - x                       # signed gap
    abs_delta = abs(delta)
    urgency   = delta / (t + 1e-6)         # how fast to close the gap
    v_sq      = v ** 2

    phi = np.array([
        x / 10.0,                           # normalised position
        np.clip(v, -20, 20) / 20.0,         # normalised velocity
        c / 10.0,                           # normalised cookie pos
        np.clip(t,  0,  5) /  5.0,         # normalised time
        np.clip(delta,    -10, 10) / 10.0,  # signed distance
        np.clip(abs_delta, 0, 10) / 10.0,   # absolute distance
        np.clip(urgency,  -10, 10) / 10.0,  # urgency
        np.clip(v_sq, 0, 400)    / 400.0,   # KE proxy
    ], dtype=np.float32)
    return phi


# ──────────────────────────────────────────────
# 2. Q-NETWORK
# ──────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    Fully-connected network that maps φ(s) → Q(s, a) for all 3 actions.

    Architecture
    ------------
    Input  : 8 features
    Hidden : 64 → 64  (ReLU activations)
    Output : 3 Q-values  (one per action: accelerate left / none / right)

    Two hidden layers are sufficient here because the feature engineering
    already handles the nonlinear work (distance, urgency).  Batch
    normalisation is intentionally omitted — it complicates target
    network updates and the state space is small.
    """
    def __init__(self, n_features: int = 8, n_actions: int = 3,
                 hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────
# 3. REPLAY BUFFER
# ──────────────────────────────────────────────

class ReplayBuffer:
    """
    Uniform experience replay buffer (FIFO).

    Stores (φ(s), a, r, φ(s'), done) tuples.
    Breaks temporal correlations in the gradient stream,
    which stabilises training significantly.
    """
    def __init__(self, capacity: int = 50_000):
        self.buf = deque(maxlen=capacity)

    def push(self, phi, a, r, phi_next, done):
        self.buf.append((phi, a, r, phi_next, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        phi, a, r, phi_next, done = zip(*batch)
        return (
            torch.FloatTensor(np.array(phi)),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(np.array(phi_next)),
            torch.FloatTensor(done),
        )

    def __len__(self):
        return len(self.buf)


# ──────────────────────────────────────────────
# 4. DQN AGENT
# ──────────────────────────────────────────────

class DQNAgent:

    def __init__(
        self,
        n_features: int = 8,
        n_actions:  int = 3,
        lr:         float = 3e-4,
        gamma:      float = 0.95,
        eps_start:  float = 1.0,
        eps_end:    float = 0.05,
        eps_decay:  float = 0.995,
        batch_size: int   = 64,
        buf_size:   int   = 50_000,
        target_sync:int   = 200,
    ):
        self.n_actions   = n_actions
        self.gamma       = gamma
        self.eps         = eps_start
        self.eps_end     = eps_end
        self.eps_decay   = eps_decay
        self.batch_size  = batch_size
        self.target_sync = target_sync
        self.steps       = 0

        self.q      = QNetwork(n_features, n_actions)
        self.q_targ = QNetwork(n_features, n_actions)
        self.q_targ.load_state_dict(self.q.state_dict())
        self.q_targ.eval()

        self.opt    = optim.Adam(self.q.parameters(), lr=lr)
        self.buf    = ReplayBuffer(buf_size)

    # ── action selection ──────────────────────────────────────────────
    def act(self, phi: np.ndarray) -> int:
        """ε-greedy policy."""
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            q_vals = self.q(torch.FloatTensor(phi).unsqueeze(0))
        return int(q_vals.argmax())

    # ── learning step ─────────────────────────────────────────────────
    def update(self) -> float | None:
        """
        One gradient update from a sampled mini-batch.
        Returns the scalar loss (for logging), or None if buffer not warm.
        """
        if len(self.buf) < self.batch_size:
            return None

        phi, a, r, phi_next, done = self.buf.sample(self.batch_size)

        # Current Q-values for chosen actions
        q_vals = self.q(phi).gather(1, a.unsqueeze(1)).squeeze(1)

        # TD target using frozen target network
        with torch.no_grad():
            q_next     = self.q_targ(phi_next).max(1).values
            td_targets = r + self.gamma * q_next * (1 - done)

        loss = nn.functional.smooth_l1_loss(q_vals, td_targets)  # Huber

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)       # grad clip
        self.opt.step()

        # Sync target network periodically
        self.steps += 1
        if self.steps % self.target_sync == 0:
            self.q_targ.load_state_dict(self.q.state_dict())

        # Decay ε
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

        return float(loss)

    def store(self, phi, a, r, phi_next, done):
        self.buf.push(phi, a, r, phi_next, done)


# ──────────────────────────────────────────────
# 5. TRAINING LOOP
# ──────────────────────────────────────────────

def train(env, n_episodes: int = 2000, max_steps: int = 1000,
          render: bool = False):
    """
    Main training loop.

    Returns
    -------
    agent         : trained DQNAgent
    returns_hist  : list of cumulative rewards per episode
    loss_hist     : list of mean losses per episode
    """
    agent = DQNAgent()
    returns_hist = []
    loss_hist    = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        phi    = extract_features(obs)

        ep_return = 0.0
        ep_losses = []

        for _ in range(max_steps):
            a   = agent.act(phi)
            obs_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            phi_next = extract_features(obs_next)
            agent.store(phi, a, r, phi_next, float(done))

            loss = agent.update()
            if loss is not None:
                ep_losses.append(loss)

            phi       = phi_next
            ep_return += r

            if done:
                break

        returns_hist.append(ep_return)
        loss_hist.append(np.mean(ep_losses) if ep_losses else float("nan"))

        if ep % 100 == 0:
            avg = np.mean(returns_hist[-100:])
            print(f"Ep {ep:4d}  return={ep_return:7.2f}  "
                  f"avg100={avg:7.2f}  ε={agent.eps:.3f}  "
                  f"loss={loss_hist[-1]:.4f}")

    return agent, returns_hist, loss_hist


# ──────────────────────────────────────────────
# 6. PLOTTING
# ──────────────────────────────────────────────

def plot_training(returns_hist, loss_hist, window: int = 50):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- Returns ---
    ax = axes[0]
    ax.plot(returns_hist, alpha=0.3, color="steelblue", label="Per-episode")
    smoothed = np.convolve(returns_hist,
                           np.ones(window) / window, mode="valid")
    ax.plot(range(window - 1, len(returns_hist)),
            smoothed, color="steelblue", label=f"{window}-ep avg")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative return")
    ax.set_title("Training returns")
    ax.legend()

    # --- Loss ---
    ax = axes[1]
    valid = [(i, l) for i, l in enumerate(loss_hist) if not np.isnan(l)]
    if valid:
        xs, ys = zip(*valid)
        ax.plot(xs, ys, alpha=0.4, color="coral")
        if len(ys) >= window:
            smooth = np.convolve(ys, np.ones(window) / window, mode="valid")
            ax.plot(xs[window - 1:], smooth, color="coral",
                    label=f"{window}-ep avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Huber loss")
    ax.set_title("TD loss")
    ax.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
    print("Saved training_curves.png")


# ──────────────────────────────────────────────
# 7. GREEDY EVALUATION
# ──────────────────────────────────────────────

def evaluate(env, agent: DQNAgent, n_episodes: int = 20,
             render: bool = False) -> float:
    """Run the greedy policy (ε=0) and return mean return."""
    agent.eps = 0.0
    returns   = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        for _ in range(1000):
            phi = extract_features(obs)
            a   = agent.act(phi)
            obs, r, terminated, truncated, _ = env.step(a)
            ep_return += r
            if terminated or truncated:
                break
        returns.append(ep_return)
    mean_r = float(np.mean(returns))
    print(f"Greedy evaluation ({n_episodes} eps): mean return = {mean_r:.2f}")
    return mean_r


# ──────────────────────────────────────────────
# 8. ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Lazy import so the file can be read without installing cookiedisaster
    from cookiedisaster.envs.cookie_disaster_env import CookieDisasterEnv

    env = CookieDisasterEnv(render_mode=None)

    print("=" * 60)
    print("Training DQN on CookieDisasterEnv")
    print("=" * 60)
    agent, returns_hist, loss_hist = train(env, n_episodes=2000)

    plot_training(returns_hist, loss_hist)
    evaluate(env, agent, n_episodes=20)
    env.close()