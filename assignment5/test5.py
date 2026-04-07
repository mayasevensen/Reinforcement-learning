"""
Exercise 5 - Task 5: Actor-Critic for CookieDisasterEnv
=======================================================

Learns a policy π(a|s) directly with an actor-critic model.

Idea
----
- Actor  : outputs logits for the 3 discrete actions
- Critic : estimates V(s)
- Update : TD(0) advantage
           δ = r + γ V(s') - V(s)

Actor loss  = -log π(a|s) * δ
Critic loss = δ²
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────
# 1. FEATURE ENGINEERING
# ──────────────────────────────────────────────

def extract_features(obs: dict) -> np.ndarray:
    """
    Same feature engineering spirit as in test.py:
    we expose the network to position, velocity, cookie location,
    timer, distance-to-cookie and urgency.
    """
    x = obs["agent"]["pos"]
    v = obs["agent"]["vel"]
    c = obs["cookie"]["pos"]
    t = obs["cookie"]["time"]

    delta = c - x
    abs_delta = abs(delta)
    urgency = delta / (t + 1e-6)
    v_sq = v ** 2

    phi = np.array([
        x / 10.0,                           # normalized position
        np.clip(v, -20, 20) / 20.0,         # normalized velocity
        c / 10.0,                           # normalized cookie position
        np.clip(t, 0, 5) / 5.0,             # normalized timer
        np.clip(delta, -10, 10) / 10.0,     # signed distance
        np.clip(abs_delta, 0, 10) / 10.0,   # absolute distance
        np.clip(urgency, -10, 10) / 10.0,   # urgency
        np.clip(v_sq, 0, 400) / 400.0,      # kinetic energy proxy
    ], dtype=np.float32)

    return phi


# ──────────────────────────────────────────────
# 2. ACTOR-CRITIC NETWORK
# ──────────────────────────────────────────────

class ActorCriticNet(nn.Module):
    """
    Shared torso + two heads:
    - actor head  -> policy logits over 3 actions
    - critic head -> scalar value V(s)
    """
    def __init__(self, n_features: int = 8, n_actions: int = 3, hidden: int = 64):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        z = self.shared(x)
        logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)
        return logits, value


# ──────────────────────────────────────────────
# 3. AGENT
# ──────────────────────────────────────────────

class ActorCriticAgent:
    def __init__(
        self,
        n_features: int = 8,
        n_actions: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        critic_coef: float = 0.5,
        grad_clip: float = 1.0,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.grad_clip = grad_clip
        self.device = torch.device(device)

        self.net = ActorCriticNet(
            n_features=n_features,
            n_actions=n_actions,
            hidden=64
        ).to(self.device)

        self.opt = optim.Adam(self.net.parameters(), lr=lr)

    def act(self, phi: np.ndarray):
        """
        Sample an action from π(a|s).
        Returns:
            action, log_prob, value, entropy
        """
        x = torch.tensor(phi, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.net(x)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return int(action.item()), log_prob, value.squeeze(0), entropy.squeeze(0)

    def greedy_act(self, phi: np.ndarray) -> int:
        """
        Greedy action for evaluation.
        """
        x = torch.tensor(phi, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.net(x)
            action = torch.argmax(logits, dim=1)
        return int(action.item())

    def update(self, phi, reward, phi_next, terminated, log_prob, value, entropy):
        """
        One-step actor-critic update using TD(0):
            δ = r + γV(s') - V(s)
        """
        x_next = torch.tensor(phi_next, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            _, next_value = self.net(x_next)
            next_value = next_value.squeeze(0)
            target = torch.tensor(reward, dtype=torch.float32, device=self.device)
            if not terminated:
                target = target + self.gamma * next_value

        advantage = target - value

        actor_loss = -(log_prob * advantage.detach())
        critic_loss = advantage.pow(2)
        entropy_bonus = entropy

        loss = actor_loss + self.critic_coef * critic_loss - self.entropy_coef * entropy_bonus

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.opt.step()

        return (
            float(loss.item()),
            float(actor_loss.item()),
            float(critic_loss.item()),
            float(advantage.item())
        )


# ──────────────────────────────────────────────
# 4. TRAINING LOOP
# ──────────────────────────────────────────────

def train_actor_critic(
    env,
    n_episodes: int = 2000,
    max_steps: int = 1000,
    render: bool = False,
):
    """
    Because CookieDisasterEnv is effectively continuing,
    we use fixed-length episodes for training.
    """
    agent = ActorCriticAgent()

    returns_hist = []
    loss_hist = []
    actor_loss_hist = []
    critic_loss_hist = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        phi = extract_features(obs)

        ep_return = 0.0
        ep_losses = []
        ep_actor_losses = []
        ep_critic_losses = []

        for _ in range(max_steps):
            action, log_prob, value, entropy = agent.act(phi)

            obs_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated   # probably always False in this env

            phi_next = extract_features(obs_next)

            loss, a_loss, c_loss, adv = agent.update(
                phi=phi,
                reward=reward,
                phi_next=phi_next,
                terminated=done,
                log_prob=log_prob,
                value=value,
                entropy=entropy,
            )

            ep_losses.append(loss)
            ep_actor_losses.append(a_loss)
            ep_critic_losses.append(c_loss)

            phi = phi_next
            ep_return += reward

            if done:
                break

        returns_hist.append(ep_return)
        loss_hist.append(np.mean(ep_losses))
        actor_loss_hist.append(np.mean(ep_actor_losses))
        critic_loss_hist.append(np.mean(ep_critic_losses))

        if ep % 100 == 0:
            avg100 = np.mean(returns_hist[-100:])
            print(
                f"Ep {ep:4d} | return={ep_return:8.2f} | "
                f"avg100={avg100:8.2f} | "
                f"loss={loss_hist[-1]:.4f} | "
                f"actor={actor_loss_hist[-1]:.4f} | "
                f"critic={critic_loss_hist[-1]:.4f}"
            )

    return agent, returns_hist, loss_hist, actor_loss_hist, critic_loss_hist


# ──────────────────────────────────────────────
# 5. EVALUATION
# ──────────────────────────────────────────────

def evaluate_actor_critic(env, agent, n_episodes: int = 20, max_steps: int = 1000):
    returns = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_return = 0.0

        for _ in range(max_steps):
            phi = extract_features(obs)
            action = agent.greedy_act(phi)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward

            if terminated or truncated:
                break

        returns.append(ep_return)

    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))
    print(f"Greedy evaluation over {n_episodes} episodes: "
          f"mean return = {mean_return:.2f} ± {std_return:.2f}")
    return mean_return, std_return


# ──────────────────────────────────────────────
# 6. PLOTTING
# ──────────────────────────────────────────────

def plot_training(returns_hist, loss_hist, actor_loss_hist, critic_loss_hist, window: int = 50):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Returns
    ax = axes[0]
    ax.plot(returns_hist, alpha=0.3, label="episode return")
    if len(returns_hist) >= window:
        smooth = np.convolve(returns_hist, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(returns_hist)), smooth, label=f"{window}-ep avg")
    ax.set_title("Training returns")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.legend()

    # Total loss
    ax = axes[1]
    ax.plot(loss_hist, alpha=0.5, label="total loss")
    if len(loss_hist) >= window:
        smooth = np.convolve(loss_hist, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(loss_hist)), smooth, label=f"{window}-ep avg")
    ax.set_title("Total loss")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.legend()

    # Actor / critic losses
    ax = axes[2]
    ax.plot(actor_loss_hist, alpha=0.6, label="actor")
    ax.plot(critic_loss_hist, alpha=0.6, label="critic")
    ax.set_title("Actor vs critic losses")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.legend()

    plt.tight_layout()
    plt.savefig("actor_critic_training.png", dpi=150)
    plt.show()
    print("Saved actor_critic_training.png")


# ──────────────────────────────────────────────
# 7. MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Adjust the import below to however you load the env locally
    from cookiedisaster.envs.cookie_disaster_env import CookieDisasterEnv

    env = CookieDisasterEnv(render_mode=None)

    print("=" * 60)
    print("Training Actor-Critic on CookieDisasterEnv")
    print("=" * 60)

    agent, returns_hist, loss_hist, actor_loss_hist, critic_loss_hist = train_actor_critic(
        env,
        n_episodes=2000,
        max_steps=1000
    )

    plot_training(returns_hist, loss_hist, actor_loss_hist, critic_loss_hist)
    evaluate_actor_critic(env, agent, n_episodes=20, max_steps=1000)

    env.close()