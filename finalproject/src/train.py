import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"

"""
Training script for the Collector DQN agent.

What changed vs the previous version:
  - Evaluation is against the BASELINE (and random), not a sampled action.
    Reports win rate and score margin -- the metrics that actually matter.
  - Best-by-eval checkpoint saved separately from the latest checkpoint.
    Late-stage degradation can no longer destroy your best agent.
  - Reward shaping is opponent-aware: the dominant signal is now
    "did my point lead grow this step?" rather than "did I get closer
    to the nearest item?". Distance shaping is kept tiny and decayed.
  - Curriculum opponent: random / baseline / past-self snapshots, mixed
    over training to prevent overfitting to baseline's quirks.
  - Per-episode reset of the agent's last-state pointer so transitions
    never cross episode boundaries.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import copy
import yaml
import numpy as np
from importlib.util import spec_from_file_location, module_from_spec

from environments.collector.wrappers import CollectorGymEnv
from environments.collector.params import EnvParams


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
with open("src/agents/agent/config.yaml") as f:
    config_dict = yaml.safe_load(f)


class Config:
    pass


config = Config()
for k, v in config_dict.items():
    setattr(config, k, v)

config.training = True
# Allow YAML to override these, but provide sane defaults if absent.
config.epsilon_start = getattr(config, 'epsilon_start', 1.0)
config.epsilon_end   = getattr(config, 'epsilon_end',   0.05)
config.epsilon_decay = getattr(config, 'epsilon_decay', 0.9985)

sys.path.insert(0, "src/agents/agent")
from agent import Agent  # noqa: E402

agent = Agent(config)
agent.load()  # picks up existing weights if present (resume training)

env = CollectorGymEnv(numpy_output=True)
env_params = EnvParams()


# ---------------------------------------------------------------------------
# Training hyperparameters (script-level, not learner-level)
# ---------------------------------------------------------------------------
NUM_EPISODES   = 4000
MAX_STEPS      = 1000
SAVE_EVERY     = 500
PRINT_EVERY    = 50
EVAL_EVERY     = 200
EVAL_GAMES     = 10        # per opponent
TRAIN_EVERY    = getattr(config, 'train_every', 4)

# Curriculum: probability of facing each opponent kind at each phase.
# (random_p, baseline_p, selfplay_p) -- must sum to 1.
def opponent_mix(episode):
    if episode < 500:
        return (0.5, 0.5, 0.0)
    elif episode < 1500:
        return (0.2, 0.7, 0.1)
    elif episode < 3000:
        return (0.1, 0.5, 0.4)
    else:
        return (0.05, 0.45, 0.5)

# How many past-self snapshots to keep in rotation.
SELFPLAY_POOL_SIZE = 3
SELFPLAY_SNAPSHOT_EVERY = 500   # episodes between snapshots


# ---------------------------------------------------------------------------
# Opponent loading helpers
# ---------------------------------------------------------------------------
def load_opponent(agent_dir):
    with open(os.path.join(agent_dir, "config.yaml")) as f:
        cfg_dict = yaml.safe_load(f)

    class Cfg:
        pass

    cfg = Cfg()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    spec = spec_from_file_location(f"opp_agent_{os.path.basename(os.path.normpath(agent_dir))}",
                                   os.path.join(agent_dir, "agent.py"))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    opp = mod.Agent(cfg)
    opp.load()
    return opp


baseline_opp = load_opponent("src/agents/baseline/")
random_opp   = load_opponent("src/agents/random/")


class FrozenSelfOpponent:
    """A frozen snapshot of our own agent, used as a self-play opponent.
    We deep-copy the network weights so training updates don't affect it."""

    def __init__(self, source_agent):
        # Build a shallow Agent shell that shares the same architecture, then
        # copy the weights. We do this by serialising/deserialising the state
        # dict to torch tensors on CPU to keep memory low.
        import torch
        # Lazily build a sibling agent with training=False
        from types import SimpleNamespace
        cfg = SimpleNamespace(**source_agent.config.__dict__) if hasattr(source_agent.config, '__dict__') else source_agent.config
        # Easier: just deep-copy the whole module. q_net is small.
        self.q_net = copy.deepcopy(source_agent.q_net).to(source_agent.device)
        self.q_net.eval()
        self.device = source_agent.device
        self._torch = torch

    def act(self, obs):
        from agent import preprocess_obs
        state = preprocess_obs(obs)
        with self._torch.no_grad():
            s = self._torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return int(self.q_net(s).argmax(dim=1).item())


selfplay_pool = []  # list of FrozenSelfOpponent


def pick_opponent(episode):
    rp, bp, sp = opponent_mix(episode)
    # Disable self-play term if pool is empty.
    if not selfplay_pool:
        bp = bp + sp
        sp = 0.0
    r = np.random.random()
    if r < rp:
        return random_opp, "random"
    elif r < rp + bp:
        return baseline_opp, "baseline"
    else:
        return np.random.choice(selfplay_pool), "selfplay"


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------
def get_nearest_item_manhattan(obs):
    raw_map = obs['map_features']['tile_type']
    my_pos = obs['units']['position'][0]
    item_locs = np.argwhere(raw_map == 2)
    if len(item_locs) == 0:
        return None
    dists = np.abs(item_locs - my_pos).sum(axis=1)
    return float(dists.min())


def shape_reward(obs, next_obs, raw_reward, distance_shape_weight):
    """
    Opponent-aware shaping.
      - Big bonus on collection (raw_reward > 0).
      - Big penalty on wall (raw_reward == -2).
      - Reward growth in our score lead vs opponent (key competitive signal).
      - Tiny optional distance shaping, decayed over training.
    """
    shaped = float(raw_reward)

    # Strong collection bonus.
    if raw_reward > 0:
        shaped += 3.0
    # Wall penalty (env already gives -2; we double its felt weight).
    if raw_reward == -2.0:
        shaped -= 1.0

    # Competitive reward: did our lead grow?
    my_pts_before  = float(obs['team_points'][0])
    opp_pts_before = float(obs['team_points'][1])
    my_pts_after   = float(next_obs['team_points'][0])
    opp_pts_after  = float(next_obs['team_points'][1])
    lead_delta = (my_pts_after - my_pts_before) - (opp_pts_after - opp_pts_before)
    shaped += 2.0 * lead_delta  # +2 if we collected, -2 if opp collected

    # Tiny distance hint, decayed away by mid-training.
    if distance_shape_weight > 0:
        d_before = get_nearest_item_manhattan(obs)
        d_after  = get_nearest_item_manhattan(next_obs)
        if d_before is not None and d_after is not None:
            shaped += distance_shape_weight * (d_before - d_after)

    return shaped


def distance_weight_schedule(episode):
    """Linearly decay from 0.2 to 0.0 over the first 1000 episodes."""
    if episode >= 1000:
        return 0.0
    return 0.2 * (1.0 - episode / 1000.0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_against(agent, opponent, env, n=10):
    """Greedy eval. Returns (mean_my_score, mean_opp_score, win_rate)."""
    old_eps = agent.epsilon
    agent.epsilon = 0.0
    my_scores, opp_scores, wins = [], [], 0
    for _ in range(n):
        obs, info = env.reset(options=dict(params=EnvParams()))
        agent.reset_episode()
        done, steps = False, 0
        while not done and steps < MAX_STEPS:
            a = agent.act(obs["player_0"])
            o = opponent.act(obs["player_1"])
            obs, _, terminated, truncated, info = env.step({"player_0": a, "player_1": o})
            done = terminated or truncated
            steps += 1
        my = int(info['state'].team_points[0])
        opp = int(info['state'].team_points[1])
        my_scores.append(my)
        opp_scores.append(opp)
        if my > opp:
            wins += 1
    agent.epsilon = old_eps
    return np.mean(my_scores), np.mean(opp_scores), wins / n


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
episode_rewards = []
episode_scores  = []
best_baseline_winrate = -1.0

print(f"Starting training. device={agent.device}, "
      f"epsilon={agent.epsilon:.3f}, hidden_dim={agent.hidden_dim}")

for episode in range(NUM_EPISODES):
    opponent, opp_name = pick_opponent(episode)
    dshape_w = distance_weight_schedule(episode)

    obs, info = env.reset(options=dict(params=env_params))
    agent.reset_episode()

    total_reward = 0.0
    done = False
    steps = 0

    while not done and steps < MAX_STEPS:
        action = agent.act(obs["player_0"])
        opp_action = opponent.act(obs["player_1"])
        actions = {"player_0": action, "player_1": opp_action}

        next_obs, reward, terminated, truncated, info = env.step(actions)

        raw_r = float(reward[0])
        shaped_r = shape_reward(obs["player_0"], next_obs["player_0"], raw_r, dshape_w)

        done = terminated or truncated
        agent.store(next_obs["player_0"], shaped_r, done)
        if steps % TRAIN_EVERY == 0:
            agent.train_step()

        obs = next_obs
        total_reward += raw_r
        steps += 1

    episode_rewards.append(total_reward)
    episode_scores.append(int(info['state'].team_points[0]))
    agent.end_episode()

    # Snapshot for self-play pool.
    if (episode + 1) % SELFPLAY_SNAPSHOT_EVERY == 0 and (episode + 1) >= 1000:
        if agent.q_net is not None:
            snap = FrozenSelfOpponent(agent)
            selfplay_pool.append(snap)
            if len(selfplay_pool) > SELFPLAY_POOL_SIZE:
                selfplay_pool.pop(0)
            print(f"  [selfplay] snapshot taken (pool size = {len(selfplay_pool)})")

    # Periodic evaluation against the real benchmarks.
    if (episode + 1) % EVAL_EVERY == 0:
        my_b, opp_b, wr_b = evaluate_against(agent, baseline_opp, env, n=EVAL_GAMES)
        my_r, opp_r, wr_r = evaluate_against(agent, random_opp,   env, n=EVAL_GAMES)
        print(f"  [eval] vs baseline: {my_b:.2f} - {opp_b:.2f}  win_rate={wr_b:.0%}  | "
              f"vs random: {my_r:.2f} - {opp_r:.2f}  win_rate={wr_r:.0%}")
        # Save best-by-baseline-winrate.
        if wr_b > best_baseline_winrate:
            best_baseline_winrate = wr_b
            agent.save(filename="weights_best.pth")
            print(f"  [eval] new best vs baseline: win_rate={wr_b:.0%} (saved as weights_best.pth)")

    if (episode + 1) % PRINT_EVERY == 0:
        avg_r = np.mean(episode_rewards[-PRINT_EVERY:])
        avg_s = np.mean(episode_scores[-PRINT_EVERY:])
        print(f"Ep {episode+1:5d} | opp={opp_name:8s} | "
              f"avg_raw_reward={avg_r:7.1f} | avg_score={avg_s:.2f} | "
              f"eps={agent.epsilon:.3f} | buf={len(agent.replay_buffer)} | "
              f"dshape={dshape_w:.2f}")

    if (episode + 1) % SAVE_EVERY == 0:
        agent.save()  # latest weights.pth
        print(f"  [save] latest weights saved (best vs baseline so far: {best_baseline_winrate:.0%})")

agent.save()
print(f"Training complete! Best vs baseline win rate: {best_baseline_winrate:.0%}")
print("Tip: copy weights_best.pth to weights.pth before submitting if best > latest.")
env.close()