from bandit import Bandits_final
import numpy as np
import matplotlib.pyplot as plt


def run_episode(agent, steps=1000):
    env = Bandits_final()
    rewards = []

    for t in range(steps):
        action = agent.select_action()
        _, reward, _, _, _ = env.step(action)
        agent.update(action, reward)
        rewards.append(reward)

    return rewards

class EpsilonGreedy:
    def __init__(self, k, epsilon=0.1):
        self.k = k
        self.epsilon = epsilon
        self.q = np.zeros(k)
        self.n = np.zeros(k)

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        return np.argmax(self.q)

    def update(self, action, reward):
        self.n[action] += 1
        self.q[action] += (reward - self.q[action]) / self.n[action]

class NonStationaryEGreedy:
    def __init__(self, k, epsilon=0.1, alpha=0.1):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha
        self.q = np.zeros(k)

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        return np.argmax(self.q)

    def update(self, action, reward):
        self.q[action] += self.alpha * (reward - self.q[action])

def experiment(agent_class, runs=50, steps=1000):
    all_rewards = np.zeros((runs, steps))

    for i in range(runs):
        agent = agent_class(3)
        rewards = run_episode(agent, steps)
        all_rewards[i] = rewards

    return np.mean(all_rewards, axis=0)


r1 = experiment(EpsilonGreedy)
r2 = experiment(NonStationaryEGreedy)

plt.plot(r1, label="Standard ε-greedy")
plt.plot(r2, label="Non-stationary ε-greedy")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.legend()
plt.show()