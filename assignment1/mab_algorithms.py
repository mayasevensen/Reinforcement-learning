import numpy as np
import matplotlib.pyplot as plt



def epsilon_greedy(mab, T, epsilon):
    k = mab.k
    E = np.zeros(k) #Ê[R|a_i] initialized to 0
    N = np.zeros(k) # initialize action counter to 0

    rewards = []
    actions = []
    

    for _ in range(T):
        if np.random.rand() < epsilon:
            action = np.random.randint(k)
        else:
            action = np.argmax(E)
        
        _, reward, _, _, _ = mab.step(action)
        rewards.append(reward)
        actions.append(action)
        N[action] += 1
        E[action] = E[action] + (1/N[action])*(reward-E[action])

    return E, N, rewards, actions


def decaying_epsilon_greedy(mab, T, epsilon, alpha):
    k = mab.k
    E = np.zeros(k) 
    N = np.zeros(k)
    rewards = []    
    actions = []

    for _ in range(T):
        if np.random.rand() < epsilon:
            action = np.random.randint(k)
        else:
            action = np.argmax(E)
        
        _, reward, _, _, _ = mab.step(action)
        rewards.append(reward)
        actions.append(action)
        N[action] += 1
        E[action] += (1/N[action])*(reward-E[action])
        
        epsilon *= alpha

    return E, N, rewards, actions


def UCB(mab, T, c):
    k = mab.k
    E = np.zeros(k) 
    N = np.zeros(k)
    U = np.zeros(k) # Confidence bound

    rewards = []
    actions = []
    
    for i in range(k):
       _, reward, _, _, _ = mab.step(i)
       rewards.append(reward)
       actions.append(i)
       N[i] += 1
       E[i] = E[i] + (1/N[i])*(reward-E[i])

    for t in range(k + 1, T+1):
        U = E + c * np.sqrt(np.log(t) / N)
        action = np.argmax(U)
        _, reward, _, _, _ = mab.step(action)
        rewards.append(reward)
        actions.append(action)

        N[action] += 1
        E[action] += (1/N[action])*(reward-E[action])

    return E, N, rewards, actions



def compute_regret(actions, true_means):
    mu_star = np.max(true_means)
    regret = []
    cumulative_regret = 0
    for a in actions:
        cumulative_regret += mu_star - true_means[a]
        regret.append(cumulative_regret)
    return regret