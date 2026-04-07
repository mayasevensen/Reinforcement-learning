"""
Exercise 5 - Q3: Neural Q-Learning for CookieDisasterEnv
========================================================
Approximates q(s,a) with a neural network and solves the control
problem via plain Q-learning.

Compared with the improved DQN version, this baseline uses:
- no replay buffer
- no target network

It updates directly from the most recent transition.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────
# 1. FEATURE ENGINEERING
# ──────────────────────────────────────────────

def extract_features(obs: dict) -> np.ndarray:
    x = obs["agent"]["pos"]
    v = obs["agent"]["vel"]
    c = obs["cookie"]["pos"]
    t = obs["cookie"]["time"]

    delta = c - x
    abs_delta = abs(delta)
    urgency = delta / (t + 1e-6)
    v_sq = v ** 2

    phi = np.array([
        x / 10.0,                           # normalised position
        np.clip(v, -20, 20) / 20.0,         # normalised velocity
        c / 10.0,                           # normalised cookie pos
        np.clip(t, 0, 5) / 5.0,             # normalised time
        np.clip(delta, -10, 10) / 10.0,     # signed distance
        np.clip(abs_delta, 0, 10) / 10.0,   # absolute distance
        np.clip(urgency, -10, 10) / 10.0,   # urgency
        np.clip(v_sq, 0, 400) / 400.0,      # KE proxy
    ], dtype=np.float32)
    return phi


# ──────────────────────────────────────────────
# 2. Q-NETWORK
# ──────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    Fully-connected network that maps φ(s) → Q(s, a) for all 3 actions.
    """
    def __init__(self, n_features: int = 8, n_actions: int = 3, hidden: int = 64):
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
# 3. NEURAL Q-LEARNING AGENT
# ──────────────────────────────────────────────

class NeuralQAgent:
    """
    Plain neural Q-learning:
    - one online network
    - no replay buffer
    - no target network
    """

    def __init__(
        self,
        n_features: int = 8,
        n_actions: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.95,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: float = 0.995,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.q = QNetwork(n_features, n_actions)
        self.opt = optim.Adam(self.q.parameters(), lr=lr)

    def act(self, phi: np.ndarray) -> int:
        """ε-greedy action selection."""
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            q_vals = self.q(torch.FloatTensor(phi).unsqueeze(0))
        return int(q_vals.argmax())

    def update(self, phi, a, r, phi_next, done) -> float:
        """
        One online Q-learning update from a single transition:
            target = r + gamma * max_a' Q(s', a')
        using the same network for both prediction and bootstrap target.
        """
        phi = torch.FloatTensor(phi).unsqueeze(0)
        phi_next = torch.FloatTensor(phi_next).unsqueeze(0)
        a = torch.LongTensor([a])
        r = torch.FloatTensor([r])
        done = torch.FloatTensor([done])

        q_val = self.q(phi).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.q(phi_next).max(1).values
            td_target = r + self.gamma * q_next * (1 - done)

        loss = nn.functional.smooth_l1_loss(q_val, td_target)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.opt.step()

        self.eps = max(self.eps_end, self.eps * self.eps_decay)

        return float(loss.item())


# ──────────────────────────────────────────────
# 4. TRAINING LOOP
# ──────────────────────────────────────────────

def train(env, n_episodes: int = 2000, max_steps: int = 1000, render: bool = False):
    """
    Main training loop.

    Returns
    -------
    agent         : trained NeuralQAgent
    returns_hist  : list of cumulative rewards per episode
    loss_hist     : list of mean losses per episode
    """
    agent = NeuralQAgent()
    returns_hist = []
    loss_hist = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        phi = extract_features(obs)

        ep_return = 0.0
        ep_losses = []

        for _ in range(max_steps):
            a = agent.act(phi)
            obs_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            phi_next = extract_features(obs_next)

            loss = agent.update(phi, a, r, phi_next, float(done))
            ep_losses.append(loss)

            phi = phi_next
            ep_return += r

            if done:
                break

        returns_hist.append(ep_return)
        loss_hist.append(np.mean(ep_losses))

        if ep % 100 == 0:
            avg = np.mean(returns_hist[-100:])
            print(
                f"Ep {ep:4d}  return={ep_return:7.2f}  "
                f"avg100={avg:7.2f}  ε={agent.eps:.3f}  "
                f"loss={loss_hist[-1]:.4f}"
            )

    return agent, returns_hist, loss_hist


# ──────────────────────────────────────────────
# 5. PLOTTING
# ──────────────────────────────────────────────

def plot_training(returns_hist, loss_hist, window: int = 50):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- Returns ---
    ax = axes[0]
    ax.plot(returns_hist, alpha=0.3, color="steelblue", label="Per-episode")
    smoothed = np.convolve(returns_hist, np.ones(window) / window, mode="valid")
    ax.plot(
        range(window - 1, len(returns_hist)),
        smoothed,
        color="steelblue",
        label=f"{window}-ep avg"
    )
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative return")
    ax.set_title("Training returns")
    ax.legend()

    # --- Loss ---
    ax = axes[1]
    ax.plot(loss_hist, alpha=0.4, color="coral")
    if len(loss_hist) >= window:
        smooth = np.convolve(loss_hist, np.ones(window) / window, mode="valid")
        ax.plot(
            range(window - 1, len(loss_hist)),
            smooth,
            color="coral",
            label=f"{window}-ep avg"
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Huber loss")
    ax.set_title("TD loss")
    ax.legend()

    plt.tight_layout()
    plt.savefig("training_curves_task3.png", dpi=150)
    plt.show()
    print("Saved training_curves_task3.png")


# ──────────────────────────────────────────────
# 6. GREEDY EVALUATION
# ──────────────────────────────────────────────

def evaluate(env, agent: NeuralQAgent, n_episodes: int = 20, render: bool = False) -> float:
    """Run the greedy policy (ε=0) and return mean return."""
    old_eps = agent.eps
    agent.eps = 0.0

    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        for _ in range(1000):
            phi = extract_features(obs)
            a = agent.act(phi)
            obs, r, terminated, truncated, _ = env.step(a)
            ep_return += r
            if terminated or truncated:
                break
        returns.append(ep_return)

    agent.eps = old_eps
    mean_r = float(np.mean(returns))
    print(f"Greedy evaluation ({n_episodes} eps): mean return = {mean_r:.2f}")
    return mean_r


# ──────────────────────────────────────────────
# 7. ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Lazy import so the file can be read without installing cookiedisaster
    from cookiedisaster.envs.cookie_disaster_env import CookieDisasterEnv

    env = CookieDisasterEnv(render_mode=None)

    print("=" * 60)
    print("Training plain neural Q-learning on CookieDisasterEnv")
    print("=" * 60)

    agent, returns_hist, loss_hist = train(env, n_episodes=2000)

    plot_training(returns_hist, loss_hist)
    evaluate(env, agent, n_episodes=20)
    env.close()