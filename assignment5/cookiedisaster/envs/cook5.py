import numpy as np
from cookiedisaster import CookieDisasterEnv
env = CookieDisasterEnv(render_mode="human")

#Feature function
def phi(obs, action):
    pos = obs["agent"]["pos"] / 10
    vel = np.tanh(obs["agent"]["vel"] / 5)
    cookie_pos = obs["cookie"]["pos"] / 10
    time_left = obs["cookie"]["time"] / 5
    distance = cookie_pos - pos

    return np.array([
        1.0,
        pos,
        vel,
        cookie_pos,
        time_left,
        distance,
        abs(distance),
        vel * distance, 
        vel**2,
        int(action == 0),
        int(action == 1),
        int(action == 2)
    ])

#q-function 
def q_hat(w, obs, action):
    return np.dot(w, phi(obs, action))


env = CookieDisasterEnv(render_mode=None)

n_features = 12
w = np.zeros(n_features)

alpha = 0.01
gamma = 0.99
epsilon = 0.1

ACTIONS = [0, 1, 2]
def choose_action(obs):
    if np.random.rand() < epsilon:
        return np.random.randint(0,3) 
    return max(ACTIONS, key=lambda a: q_hat(w, obs, a))

episode_rewards = []
for episode in range(1000):
    epsilon = max(0.01, epsilon * 0.995)
    obs, _ = env.reset()
    total_reward = 0

    for t in range(500):
        action = choose_action(obs)
        next_obs, reward, _, _, _ = env.step(action)
        total_reward += reward

        q_current = q_hat(w, obs, action)
        q_next = max(q_hat(w, next_obs, a) for a in ACTIONS)
        q_next = np.clip(q_next, -5, 5)
        delta = reward + gamma * q_next - q_current

        w += alpha * delta * phi(obs, action)

        obs = next_obs

    if episode % 100 == 0:
        print(f"Episode {episode}, reward: {total_reward}")
    episode_rewards.append(total_reward)


obs, _ = env.reset()

total_reward = 0
for t in range(500):
    action = max(ACTIONS, key=lambda a: q_hat(w, obs, a))
    obs, reward, _, _, _ = env.step(action)
    total_reward += reward

print("Test reward:", total_reward)