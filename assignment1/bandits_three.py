from mab_algorithms import epsilon_greedy, decaying_epsilon_greedy, UCB, compute_regret
from bandit import Bandits_three
import numpy as np
import matplotlib.pyplot as plt


n_runs = 20
T = 1000

# epsilon greedy
epsilon_E = []
epsilon_N = []
epsilon_rewards = []
epsilon_actions = []

decaying_E = []
decaying_N = []
decaying_rewards = []
decaying_actions = []

UCB_E = []
UCB_N = []
UCB_rewards = []
UCB_actions = []

for i in range(n_runs):
    mab = Bandits_three(random_state = i)
    
    E, N, rewards, actions = epsilon_greedy(mab, T, 0.1)
    epsilon_E.append(E)
    epsilon_N.append(N)
    epsilon_rewards.append(rewards)
    epsilon_actions.append(actions)

    E, N, rewards, actions = decaying_epsilon_greedy(mab, T, 0.99, 0.99)
    decaying_E.append(E)   
    decaying_N.append(N)
    decaying_rewards.append(rewards)
    decaying_actions.append(actions)

    E, N, rewards, actions = UCB(mab, T, 2)
    UCB_E.append(E)
    UCB_N.append(N)
    UCB_rewards.append(rewards)
    UCB_actions.append(actions)

# Calculate average rewards and optimal action percentages
avg_epsilon_E = np.mean(epsilon_E, axis=0)
avg_epsilon_N = np.mean(epsilon_N, axis=0)

avg_decaying_E = np.mean(decaying_E, axis=0)
avg_decaying_N = np.mean(decaying_N, axis=0)

avg_UCB_E = np.mean(UCB_E, axis=0)
avg_UCB_N = np.mean(UCB_N, axis=0)

print("Epsilon Greedy Average Reward:", avg_epsilon_E)
print("Decaying Epsilon Greedy Average Reward:", avg_decaying_E)
print("UCB Average Reward:", avg_UCB_E)

# print N exploration counts
print("Epsilon Greedy Average Action Counts:", avg_epsilon_N)
print("Decaying Epsilon Greedy Average Action Counts:", avg_decaying_N)
print("UCB Average Action Counts:", avg_UCB_N)



epsilon_regret = np.zeros((n_runs, T))
decaying_epsilon_regret = np.zeros((n_runs, T))
UCB_regret = np.zeros((n_runs, T))

for i in range(n_runs):
    mab = Bandits_three(random_state=i)
    true_means = mab.means

    epsilon_regret[i] = compute_regret(epsilon_actions[i], true_means)
    decaying_epsilon_regret[i] = compute_regret(decaying_actions[i], true_means)
    UCB_regret[i] = compute_regret(UCB_actions[i], true_means)
    

epsilon_greedy_avg_regret = np.mean(epsilon_regret, axis=0)
decaying_avg_regret = np.mean(decaying_epsilon_regret, axis=0)
UCB_avg_regret = np.mean(UCB_regret, axis=0)

# Plotting the average regret
plt.figure(figsize=(10, 6))
x = np.arange(1, T + 1)
plt.plot(x, epsilon_greedy_avg_regret, label='Epsilon Greedy',)
plt.plot(x, decaying_avg_regret, label='Decaying Epsilon Greedy')
plt.plot(x, UCB_avg_regret, label='UCB')
plt.xlabel('Time Steps')
plt.ylabel('Average Regret')
plt.title('Average Regret over Time for Different MAB Algorithms')
plt.legend()
plt.show()
