from mab_algorithms import epsilon_greedy, decaying_epsilon_greedy, UCB
from bandit import Bandits_four
import numpy as np

n_runs = 20
T = 1000

algorithms = ["epsilon", "decaying", "ucb"]

results = {
    gene: {
        alg: {"E": [], "N": [], "actions": [], "rewards": []}
        for alg in algorithms
    }
    for gene in [0, 1]
}

for i in range(n_runs):
    for gene in [0, 1]:

        # ε-greedy
        mab = Bandits_four(random_state=i, gene=gene)
        E, N, rewards, actions = epsilon_greedy(mab, T, epsilon=0.1)
        results[gene]["epsilon"]["E"].append(E)
        results[gene]["epsilon"]["N"].append(N)
        results[gene]["epsilon"]["rewards"].append(rewards)
        results[gene]["epsilon"]["actions"].append(actions)

        # Decaying ε-greedy
        mab = Bandits_four(random_state=i, gene=gene)
        E, N, rewards, actions = decaying_epsilon_greedy(mab, T, epsilon=0.99, alpha=0.99)
        results[gene]["decaying"]["E"].append(E)
        results[gene]["decaying"]["N"].append(N)
        results[gene]["decaying"]["rewards"].append(rewards)
        results[gene]["decaying"]["actions"].append(actions)

        # UCB
        mab = Bandits_four(random_state=i, gene=gene)
        E, N, rewards, actions = UCB(mab, T, c=2)
        results[gene]["ucb"]["E"].append(E)
        results[gene]["ucb"]["N"].append(N)
        results[gene]["ucb"]["rewards"].append(rewards)
        results[gene]["ucb"]["actions"].append(actions)


for gene in [0, 1]:
    print(f"\n===== Gene G = {gene} =====")

    for alg in algorithms:
        avg_E = np.mean(results[gene][alg]["E"], axis=0)
        avg_N = np.mean(results[gene][alg]["N"], axis=0)

        print(f"\n{alg}")
        print("Average estimated rewards:", avg_E)
        print("Average action counts:", avg_N)
