import numpy as np
from cookiedisaster import CookieDisasterEnv
env = CookieDisasterEnv(render_mode="human")

#Feature function
def phi(obs, action):
    pos = obs["agent"]["pos"]
    vel = obs["agent"]["vel"]
    cookie_pos = obs["cookie"]["pos"]
    time_left = obs["cookie"]["time"]

    distance = cookie_pos - pos

    return np.array([
        pos,
        vel,
        cookie_pos,
        time_left,
        distance,
        vel**2,
        int(action == 0),
        int(action == 1),
        int(action == 2)
    ])

#q-function 
def q_hat(w, obs, action):
    return np.dot(w, phi(obs, action))


env = CookieDisasterEnv(render_mode=None)

n_features = 9
w = np.zeros(n_features)

alpha = 0.01
gamma = 0.99
epsilon = 0.1

ACTIONS = [0, 1, 2]
def choose_action(obs):
    if np.random.rand() < epsilon:
        return np.random.choice(ACTIONS)
    return max(ACTIONS, key=lambda a: q_hat(w, obs, a))

for episode in range(200):
    obs, _ = env.reset()

    for t in range(500):
        action = choose_action(obs)
        next_obs, reward, _, _, _ = env.step(action)

        q_current = q_hat(w, obs, action)
        q_next = max(q_hat(w, next_obs, a) for a in ACTIONS)

        delta = reward + gamma * q_next - q_current

        w += alpha * delta * phi(obs, action)

        obs = next_obs