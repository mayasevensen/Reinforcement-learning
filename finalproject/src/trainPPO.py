"""
Training script for the PPO agent.

Key difference from trainDQN.py:
  PPO is ON-POLICY. Instead of a replay buffer with random sampling, we:
    1. Run the current policy for ROLLOUT_LEN steps (collecting a trajectory).
    2. Compute GAE advantages over the whole rollout.
    3. Run PPO_EPOCHS passes of minibatch SGD on that rollout.
    4. Discard the rollout and repeat.

This means we call agent.train_on_rollout() every ROLLOUT_LEN steps,
not agent.train_step() every few steps.

Curriculum (identical phases to trainDQN.py for fair comparison):
  Phase 1 (ep    0– 300): Passive opponent
  Phase 2 (ep  300– 800): Random
  Phase 3 (ep  800–1800): Random -> Baseline
  Phase 4 (ep 1800–3500): Baseline -> BFS
  Phase 5 (ep 3500–5000): BFS + Selfplay
"""

import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import copy
import yaml
import builtins
import numpy as np
from importlib.util import spec_from_file_location, module_from_spec

from environments.collector.wrappers import CollectorGymEnv
from environments.collector.params import EnvParams


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs("plots", exist_ok=True)
LOG_PATH  = "plots/training_log_ppo.txt"
_log_file = open(LOG_PATH, "w", buffering=1)
_orig_print = builtins.print

def _tee_print(*args, **kwargs):
    _orig_print(*args, **kwargs)
    kwargs.pop("file", None)
    _orig_print(*args, file=_log_file, **kwargs)

builtins.print = _tee_print
import atexit
atexit.register(_log_file.close)
print(f"[log] writing PPO training log to {os.path.abspath(LOG_PATH)}")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
with open("src/agents/agent_ppo/config.yaml") as f:
    config_dict = yaml.safe_load(f)

class Config:
    pass

config = Config()
for k, v in config_dict.items():
    setattr(config, k, v)

config.training = True

sys.path.insert(0, "src/agents/agent_ppo")
from agent import Agent, preprocess_obs  # noqa: E402

agent = Agent(config)
agent.load()

env        = CollectorGymEnv(numpy_output=True)
env_params = EnvParams()


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
NUM_EPISODES  = 5000
MAX_STEPS     = 1000
ROLLOUT_LEN   = getattr(config, 'rollout_len', 512)
SAVE_EVERY    = 500
PRINT_EVERY   = 50
EVAL_EVERY    = 200
EVAL_GAMES    = 10

SELFPLAY_POOL_SIZE      = 5
SELFPLAY_SNAPSHOT_EVERY = 300
SELFPLAY_START_EP       = 2000


# ---------------------------------------------------------------------------
# Passive opponent
# ---------------------------------------------------------------------------
class PassiveOpponent:
    def act(self, obs):
        return 0


# ---------------------------------------------------------------------------
# Opponent loading
# ---------------------------------------------------------------------------
def load_opponent(agent_dir):
    with open(os.path.join(agent_dir, "config.yaml")) as f:
        cfg_dict = yaml.safe_load(f)

    class Cfg:
        pass
    cfg = Cfg()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)

    module_name = f"opp_{os.path.basename(os.path.normpath(agent_dir))}"
    spec = spec_from_file_location(module_name, os.path.join(agent_dir, "agent.py"))
    mod  = module_from_spec(spec)
    spec.loader.exec_module(mod)
    opp = mod.Agent(cfg)
    opp.load()
    return opp


passive_opp  = PassiveOpponent()
random_opp   = load_opponent("src/agents/random/")
baseline_opp = load_opponent("src/agents/baseline/")
bfs_opp      = load_opponent("src/agents/bfs/")
print("[init] opponents loaded: passive, random, baseline, bfs")


# ---------------------------------------------------------------------------
# Frozen self-play opponent
# ---------------------------------------------------------------------------
class FrozenSelfOpponent:
    def __init__(self, source_agent):
        import torch
        self.net    = copy.deepcopy(source_agent.net).to(source_agent.device)
        self.net.eval()
        self.device = source_agent.device
        self._torch = torch

    def act(self, obs):
        state = preprocess_obs(obs)
        s     = self._torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            logits, _ = self.net(s)
        return int(logits.argmax(dim=1).item())


selfplay_pool = []


# ---------------------------------------------------------------------------
# Curriculum (identical to DQN training for fair comparison)
# ---------------------------------------------------------------------------
def opponent_mix(episode):
    if episode < 300:
        return (1.0, 0.0, 0.0, 0.0, 0.0)
    elif episode < 800:
        return (0.0, 1.0, 0.0, 0.0, 0.0)
    elif episode < 1800:
        t  = (episode - 800) / 1000.0
        rp = max(0.0, 0.4 - 0.3 * t)
        bp = min(0.9, 0.6 + 0.3 * t)
        return (0.0, rp, bp, 0.0, 0.0)
    elif episode < 3500:
        t    = (episode - 1800) / 1700.0
        bp   = max(0.1, 0.6 - 0.5 * t)
        bfsp = min(0.9, 0.4 + 0.5 * t)
        return (0.0, 0.0, bp, bfsp, 0.0)
    else:
        sp   = 0.4 if selfplay_pool else 0.0
        bfsp = 1.0 - sp
        return (0.0, 0.0, 0.0, bfsp, sp)


def pick_opponent(episode):
    pp, rp, bp, bfsp, sp = opponent_mix(episode)
    if not selfplay_pool and sp > 0:
        bfsp += sp
        sp    = 0.0
    r = np.random.random()
    cumulative = 0.0
    for prob, opp, name in [
        (pp,   passive_opp,  "passive"),
        (rp,   random_opp,   "random"),
        (bp,   baseline_opp, "baseline"),
        (bfsp, bfs_opp,      "bfs"),
        (sp,   None,         "selfplay"),
    ]:
        cumulative += prob
        if r < cumulative:
            if name == "selfplay":
                return np.random.choice(selfplay_pool), "selfplay"
            return opp, name
    return bfs_opp, "bfs"


# ---------------------------------------------------------------------------
# Reward shaping (identical to DQN training)
# ---------------------------------------------------------------------------
def get_nearest_item_manhattan(obs):
    raw_map   = obs['map_features']['tile_type']
    my_pos    = obs['units']['position'][0]
    item_locs = np.argwhere(raw_map == 2)
    if len(item_locs) == 0:
        return None
    return float(np.abs(item_locs - my_pos).sum(axis=1).min())


def shape_reward(obs, next_obs, raw_reward, distance_shape_weight):
    shaped = float(raw_reward)
    if raw_reward > 0:
        shaped += 3.0
    if raw_reward == -2.0:
        shaped -= 1.0

    my_pts_before  = float(obs['team_points'][0])
    opp_pts_before = float(obs['team_points'][1])
    my_pts_after   = float(next_obs['team_points'][0])
    opp_pts_after  = float(next_obs['team_points'][1])
    lead_delta     = (my_pts_after - my_pts_before) - (opp_pts_after - opp_pts_before)
    shaped        += 1.5 * lead_delta

    if distance_shape_weight > 0:
        d_before = get_nearest_item_manhattan(obs)
        d_after  = get_nearest_item_manhattan(next_obs)
        if d_before is not None and d_after is not None:
            shaped += distance_shape_weight * (d_before - d_after)

    return shaped


def distance_weight_schedule(episode):
    if episode >= 800:
        return 0.0
    return 0.2 * (1.0 - episode / 800.0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_against(agent, opponent, env, n=10):
    my_scores, opp_scores, wins = [], [], 0
    for _ in range(n):
        obs, info = env.reset(options=dict(params=EnvParams()))
        agent.reset_episode()
        done, steps = False, 0
        while not done and steps < MAX_STEPS:
            a = agent.act(obs["player_0"])
            o = opponent.act(obs["player_1"])
            obs, _, terminated, truncated, info = env.step({"player_0": a, "player_1": o})
            done  = terminated or truncated
            steps += 1
        my  = int(info['state'].team_points[0])
        opp = int(info['state'].team_points[1])
        my_scores.append(my)
        opp_scores.append(opp)
        if my > opp:
            wins += 1
    return np.mean(my_scores), np.mean(opp_scores), wins / n


def checkpoint_score(wr_baseline, my_bfs, opp_bfs):
    bfs_margin = (my_bfs - opp_bfs) / max(my_bfs + opp_bfs, 1.0)
    return wr_baseline + 0.2 * bfs_margin


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
episode_rewards = []
episode_scores  = []
best_ckpt_score = -999.0

# Steps-since-last-update counter (PPO updates on rollout length, not episode)
steps_since_update = 0

print(f"Starting PPO training. device={agent.device}, "
      f"hidden_dim={agent.hidden_dim}, rollout_len={ROLLOUT_LEN}, "
      f"ppo_epochs={agent.ppo_epochs}, num_episodes={NUM_EPISODES}")
print("Phases: 0-300 passive | 300-800 random | 800-1800 baseline | "
      "1800-3500 BFS | 3500-5000 BFS+selfplay")

for episode in range(NUM_EPISODES):
    opponent, opp_name = pick_opponent(episode)
    dshape_w           = distance_weight_schedule(episode)

    obs, info = env.reset(options=dict(params=env_params))
    agent.reset_episode()

    total_reward = 0.0
    done         = False
    steps        = 0
    last_obs_for_bootstrap = obs["player_0"]   # fallback bootstrap obs

    while not done and steps < MAX_STEPS:
        action     = agent.act(obs["player_0"])
        opp_action = opponent.act(obs["player_1"])

        next_obs, reward, terminated, truncated, info = env.step(
            {"player_0": action, "player_1": opp_action}
        )

        raw_r    = float(reward[0])
        shaped_r = shape_reward(obs["player_0"], next_obs["player_0"], raw_r, dshape_w)
        done     = terminated or truncated

        agent.store(next_obs["player_0"], shaped_r, done)
        steps_since_update        += 1
        last_obs_for_bootstrap     = next_obs["player_0"]

        # ------------------------------------------------------------------
        # PPO update: triggered when rollout buffer is full, NOT every step.
        # We bootstrap with V(last_obs) to handle mid-episode updates.
        # ------------------------------------------------------------------
        if steps_since_update >= ROLLOUT_LEN:
            agent.train_on_rollout(last_obs_for_bootstrap)
            steps_since_update = 0

        obs          = next_obs
        total_reward += raw_r
        steps        += 1

    # If episode ended before rollout was full, still update so we don't
    # accumulate stale data across episode boundaries.
    if steps_since_update > 0 and len(agent.buffer) > 0:
        agent.train_on_rollout(last_obs_for_bootstrap)
        steps_since_update = 0

    episode_rewards.append(total_reward)
    episode_scores.append(int(info['state'].team_points[0]))

    # ------------------------------------------------------------------
    # Selfplay snapshot
    # ------------------------------------------------------------------
    if (episode + 1) >= SELFPLAY_START_EP and \
       (episode + 1) % SELFPLAY_SNAPSHOT_EVERY == 0 and \
       agent.net is not None:
        snap = FrozenSelfOpponent(agent)
        selfplay_pool.append(snap)
        if len(selfplay_pool) > SELFPLAY_POOL_SIZE:
            selfplay_pool.pop(0)
        print(f"  [selfplay] snapshot at ep {episode+1} "
              f"(pool={len(selfplay_pool)})")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    if (episode + 1) % EVAL_EVERY == 0:
        my_b,   opp_b,   wr_b   = evaluate_against(agent, baseline_opp, env, n=EVAL_GAMES)
        my_r,   opp_r,   wr_r   = evaluate_against(agent, random_opp,   env, n=EVAL_GAMES)
        my_bfs, opp_bfs, wr_bfs = evaluate_against(agent, bfs_opp,      env, n=EVAL_GAMES)

        print(f"  [eval ep {episode+1}] "
              f"vs baseline: {my_b:.1f}-{opp_b:.1f} wr={wr_b:.0%} | "
              f"vs random: {my_r:.1f}-{opp_r:.1f} wr={wr_r:.0%} | "
              f"vs bfs: {my_bfs:.1f}-{opp_bfs:.1f} wr={wr_bfs:.0%}")

        score = checkpoint_score(wr_b, my_bfs, opp_bfs)
        if score > best_ckpt_score:
            best_ckpt_score = score
            agent.save(filename="weights_ppo_best.pth")
            print(f"  [eval] new best PPO checkpoint (score={score:.3f}, "
                  f"baseline_wr={wr_b:.0%}, bfs_margin={my_bfs-opp_bfs:.1f})")

    # ------------------------------------------------------------------
    # Periodic print
    # ------------------------------------------------------------------
    if (episode + 1) % PRINT_EVERY == 0:
        avg_r = np.mean(episode_rewards[-PRINT_EVERY:])
        avg_s = np.mean(episode_scores[-PRINT_EVERY:])
        phase = (
            "passive"  if episode < 300  else
            "random"   if episode < 800  else
            "baseline" if episode < 1800 else
            "bfs"      if episode < 3500 else
            "bfs+self"
        )
        print(f"Ep {episode+1:5d} [{phase:8s}] opp={opp_name:8s} | "
              f"avg_reward={avg_r:7.1f} | avg_score={avg_s:.2f} | "
              f"buf={len(agent.buffer)}")

    # ------------------------------------------------------------------
    # Periodic save
    # ------------------------------------------------------------------
    if (episode + 1) % SAVE_EVERY == 0:
        agent.save()
        print(f"  [save] ep {episode+1}: weights_ppo.pth saved "
              f"(best score: {best_ckpt_score:.3f})")

agent.save()
print(f"\nPPO training complete after {NUM_EPISODES} episodes.")
print(f"Best checkpoint score: {best_ckpt_score:.3f}")
print("Tip: use weights_ppo_best.pth if it outperforms weights_ppo.pth.")
env.close()