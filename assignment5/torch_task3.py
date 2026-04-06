import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib as plt

def state_to_vector(obs):
    x = obs["agent"]["pos"]
    v = obs["agent"]["vel"]
    c = obs["cookie"]["pos"]
    t = obs["cookie"]["time"]

    # simple normalization
    x_n = x / 10.0
    c_n = c / 10.0
    t_n = t / 5.0
    v_n = np.clip(v / 5.0, -2.0, 2.0)

    d = c - x
    d_n = d / 10.0
    abs_d_n = abs(d) / 10.0
    v2_n = (v_n ** 2)

    return np.array([x_n, v_n, c_n, t_n, d_n, abs_d_n, v2_n], dtype=np.float32)

class QNetwork(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
    
class NeuralQAgent:
    def __init__(self, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork().to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state_vec):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 3)

        with torch.no_grad():
            s = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(s)
            return int(torch.argmax(q_values, dim=1).item())

    def train_step(self, state_vec, action, reward, next_state_vec, done):
        s = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        s_next = torch.tensor(next_state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)

        q_values = self.q_net(s)
        q_sa = q_values[0, action]

        with torch.no_grad():
            next_q_values = self.q_net(s_next)
            max_next_q = torch.max(next_q_values, dim=1)[0].item()
            target = reward if done else reward + self.gamma * max_next_q

        target_tensor = torch.tensor(target, dtype=torch.float32, device=self.device)

        loss = self.loss_fn(q_sa, target_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

env = CookieDisasterEnv(render_mode=None)
agent = NeuralQAgent(lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995)

num_episodes = 500
max_steps = 200

returns = []
losses = []

for ep in range(num_episodes):
    obs, info = env.reset()
    state_vec = state_to_vector(obs)

    total_reward = 0.0
    episode_losses = []

    for step in range(max_steps):
        action = agent.select_action(state_vec)
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state_vec = state_to_vector(next_obs)

        done = terminated or truncated
        loss = agent.train_step(state_vec, action, reward, next_state_vec, done)

        state_vec = next_state_vec
        total_reward += reward
        episode_losses.append(loss)

        if done:
            break

    agent.decay_epsilon()
    returns.append(total_reward)
    losses.append(np.mean(episode_losses))

    if (ep + 1) % 50 == 0:
        print(f"Episode {ep+1}, avg return (last 50): {np.mean(returns[-50:]):.3f}, avg loss: {np.mean(losses[-50:]):.4f}")

env.close()


plt.figure(figsize=(8,5))
plt.plot(returns, alpha=0.6, label="Episode return")

window = 20
moving_avg = np.convolve(returns, np.ones(window)/window, mode='valid')
plt.plot(range(window-1, len(returns)), moving_avg, label="Moving average")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("Neural Network Q-learning Performance")
plt.legend()
plt.show()