import numpy as np
from bandit import Bandits_one, Bandits_two
import matplotlib.pyplot as plt
from mab_algorithms import epsilon_greedy, decaying_epsilon_greedy, UCB, compute_regret

T = 1000
n_runs = 20



mab1 = Bandits_one()

E, N, _, _ = epsilon_greedy(mab1, T, 0.1)
print(E)
print(N)
E, N, _, _ = decaying_epsilon_greedy(mab1, T, 0.99, 0.99)
print(E)
print(N)
E, N, _, _ = UCB(mab1, T, 2)
print(E)
print(N)


epsilon_greedy_E = np.zeros(mab1.k)
epsilon_greedy_N = np.zeros(mab1.k)
epsilon_greedy_rewards = []
epsilon_greedy_actions = []

decaying_epsilon_greedy_E = np.zeros(mab1.k)
decaying_epsilon_greedy_N = np.zeros(mab1.k)
decaying_epsilon_greedy_rewards = []
decaying_epsilon_greedy_actions = []

UCB_E = np.zeros(mab1.k)
UCB_N = np.zeros(mab1.k)
UCB_rewards = []
UCB_actions = []


for i in range(n_runs):
    mab = Bandits_one(random_state=i)

    E, N, reward, actions = epsilon_greedy(mab, T, 0.1)
    epsilon_greedy_E += E
    epsilon_greedy_N += N
    epsilon_greedy_rewards.append(reward)
    epsilon_greedy_actions.append(actions)

    mab = Bandits_one(random_state=i)
    E, N, reward, actions = decaying_epsilon_greedy(mab, T, 0.99, 0.99)
    decaying_epsilon_greedy_E += E
    decaying_epsilon_greedy_N += N
    decaying_epsilon_greedy_rewards.append(reward)
    decaying_epsilon_greedy_actions.append(actions)

    mab = Bandits_one(random_state=i)
    E, N, reward, actions = UCB(mab, T, 2)
    UCB_E += E
    UCB_N += N
    UCB_rewards.append(reward)
    UCB_actions.append(actions)

    

epsilon_greedy_E /= n_runs
epsilon_greedy_N /= n_runs

decaying_epsilon_greedy_E /= n_runs
decaying_epsilon_greedy_N /= n_runs

UCB_E /= n_runs
UCB_N /= n_runs

print("Epsilon Greedy Average Estimated Rewards:", epsilon_greedy_E)
print("Epsilon Greedy Average Action Counts:", epsilon_greedy_N)
print()
print("Decaying Epsilon Greedy Average Estimated Rewards:", decaying_epsilon_greedy_E)
print("Decaying Epsilon Greedy Average Action Counts:", decaying_epsilon_greedy_N)
print()
print("UCB Average Estimated Rewards:", UCB_E)
print("UCB Average Action Counts:", UCB_N)


# epsilon greedy regret
epsilon_regret = np.zeros((n_runs, T))
decaying_epsilon_regret = np.zeros((n_runs, T))
UCB_regret = np.zeros((n_runs, T))

for i in range(n_runs):
    mab = Bandits_one(random_state=i)
    true_means = mab.means

    epsilon_regret[i] = compute_regret(epsilon_greedy_actions[i], true_means)
    decaying_epsilon_regret[i] = compute_regret(decaying_epsilon_greedy_actions[i], true_means)
    UCB_regret[i] = compute_regret(UCB_actions[i], true_means)
    

epsilon_greedy_avg_regret = np.mean(epsilon_regret, axis=0)
decaying_avg_regret = np.mean(decaying_epsilon_regret, axis=0)
UCB_avg_regret = np.mean(UCB_regret, axis=0)

# Plotting the average regret

x = np.arange(1, T + 1)

plt.figure(figsize=(10, 6))

plt.plot(x, epsilon_greedy_avg_regret, label="ε-greedy (ε=0.1)")
plt.plot(x, decaying_avg_regret, label="Decaying ε-greedy")
plt.plot(x, UCB_avg_regret, label="UCB (c=2)")

plt.xlabel("Time step")
plt.ylabel("Regret")
plt.title("Average Regret over Time")
plt.legend()
plt.grid(True)
plt.show()

