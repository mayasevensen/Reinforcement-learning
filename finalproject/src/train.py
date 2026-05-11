"""
Training script for the Collector DQN agent.

Curriculum (5 phases over 5000 episodes):
  Phase 1 (ep    0– 300): Passive opponent (stands still).
                           Agent learns the basic collect-items signal
                           before any competition is introduced.
  Phase 2 (ep  300– 800): vs Random only.
                           Introduces movement noise and mild competition.
  Phase 3 (ep  800–1600): vs Random + Baseline (weighted toward baseline).
                           Baseline's greedy Manhattan strategy is a real threat.
  Phase 4 (ep 1600–3500): vs Baseline + BFS (weighted toward BFS).
                           BFS routes around obstacles perfectly — forces the
                           agent to learn proper competitive navigation.
  Phase 5 (ep 3500–5000): vs BFS + Selfplay.
                           Selfplay against frozen past-self snapshots teaches
                           the agent to handle adaptive opponents (like other
                           student agents in the tournament).

Other changes vs previous train.py:
  - PassiveOpponent added for Phase 1.
  - BFS opponent loaded from src/agents/bfs/.
  - FrozenSelfOpponent no longer imports preprocess_obs at call-time
    (avoids module-path fragility); uses a bound reference instead.
  - Selfplay snapshots start at episode 2000 (agent is decent by then).
  - Selfplay snapshot frequency increased to every 300 episodes.
  - Eval now also reports vs BFS so we can track progress on the hard opponent.
  - Best checkpoint tracks best combined score (baseline win_rate + BFS margin)
    rather than baseline win_rate alone, which could plateau early.
  - Reward shaping: lead_delta weight reduced from 2.0 -> 1.5 to reduce
    double-counting with the raw collection bonus (+1 env + 3.0 shaped).
  - Distance shaping decayed over first 800 episodes (was 1000) to align
    with the earlier transition away from solo/random play.
"""

# --- Suppress noisy library warnings BEFORE any other imports ---
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
# Tee print(): every print() call also writes to training_log.txt.
# The log file is overwritten at the start of each training run.
# ---------------------------------------------------------------------------
os.makedirs("plots", exist_ok=True)
LOG_PATH = "plots/training_log.txt"
_log_file = open(LOG_PATH, "w", buffering=1)  # line-buffered: flushes per print
_orig_print = builtins.print

def _tee_print(*args, **kwargs):
    _orig_print(*args, **kwargs)
    kwargs.pop("file", None)
    _orig_print(*args, file=_log_file, **kwargs)

builtins.print = _tee_print
import atexit
atexit.register(_log_file.close)
print(f"[log] writing training log to {os.path.abspath(LOG_PATH)}")


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
config.epsilon_start = getattr(config, 'epsilon_start', 1.0)
config.epsilon_end   = getattr(config, 'epsilon_end',   0.05)
config.epsilon_decay = getattr(config, 'epsilon_decay', 0.9985)

sys.path.insert(0, "src/agents/agent")
from agent import Agent, preprocess_obs  # noqa: E402

agent = Agent(config)
agent.load()  # resume from existing weights if present

env = CollectorGymEnv(numpy_output=True)
env_params = EnvParams()


# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
NUM_EPISODES    = 5000
MIN_TRAIN_STEPS = 200   # competition games are mostly 200-600 steps
MAX_TRAIN_STEPS = 800   # episode length sampled uniformly in this range
SAVE_EVERY    = 500
PRINT_EVERY   = 50
EVAL_EVERY    = 200
EVAL_GAMES    = 10       # games per opponent during eval
TRAIN_EVERY   = getattr(config, 'train_every', 4)

SELFPLAY_POOL_SIZE      = 5    # keep the 5 most recent snapshots
SELFPLAY_SNAPSHOT_EVERY = 300  # episodes between snapshots
SELFPLAY_START_EP       = 2000 # don't snapshot until agent is decent


# ---------------------------------------------------------------------------
# Passive opponent: always plays action 0 (move up / into wall).
# Used in Phase 1 so the agent learns item collection with no competition.
# ---------------------------------------------------------------------------
class PassiveOpponent:
    """Does nothing useful — just repeatedly tries to move up."""
    def act(self, obs):
        return 0  # UP into wall every step; effectively stationary


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

    module_name = f"opp_agent_{os.path.basename(os.path.normpath(agent_dir))}"
    spec = spec_from_file_location(module_name, os.path.join(agent_dir, "agent.py"))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    opp = mod.Agent(cfg)
    opp.load()
    return opp


passive_opp  = PassiveOpponent()
random_opp   = load_opponent("src/agents/random/")
baseline_opp = load_opponent("src/agents/baseline/")
bfs_opp      = load_opponent("src/agents/bfs/")

print("[init] all opponents loaded: passive, random, baseline, bfs")


# ---------------------------------------------------------------------------
# Frozen self-play opponent
# Uses a bound reference to preprocess_obs imported above, so there's no
# fragile module re-import at call time.
# ---------------------------------------------------------------------------
class FrozenSelfOpponent:
    """Frozen snapshot of our agent. Training updates don't affect it."""

    def __init__(self, source_agent):
        import torch
        self.q_net  = copy.deepcopy(source_agent.q_net).to(source_agent.device)
        self.q_net.eval()
        self.device = source_agent.device
        self._torch = torch

    def act(self, obs):
        state = preprocess_obs(obs)   # bound at import time — no re-import
        with self._torch.no_grad():
            s = self._torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return int(self.q_net(s).argmax(dim=1).item())


selfplay_pool = []


# ---------------------------------------------------------------------------
# Curriculum: opponent mix per phase
# Returns (passive_p, random_p, baseline_p, bfs_p, selfplay_p)
# ---------------------------------------------------------------------------
def opponent_mix(episode):
    if episode < 300:
        # Phase 1: solo learning — passive opponent only
        return (1.0, 0.0, 0.0, 0.0, 0.0)
    elif episode < 800:
        # Phase 2: introduce random competition
        return (0.0, 1.0, 0.0, 0.0, 0.0)
    elif episode < 1600:
        # Phase 3: ramp up baseline
        t = (episode - 800) / 1000.0          # 0 -> 1 over this phase
        rp = max(0.0, 0.4 - 0.3 * t)         # 0.4 -> 0.1
        bp = min(0.9, 0.6 + 0.3 * t)         # 0.6 -> 0.9
        return (0.0, rp, bp, 0.0, 0.0)
    elif episode < 3500:
        # Phase 4: introduce BFS, fade out baseline
        t = (episode - 1600) / 1700.0         # 0 -> 1 over this phase
        bp = max(0.1, 0.6 - 0.5 * t)         # 0.6 -> 0.1
        bfsp = min(0.9, 0.4 + 0.5 * t)       # 0.4 -> 0.9
        return (0.0, 0.0, bp, bfsp, 0.0)
    else:
        # Phase 5: BFS + selfplay (selfplay grows if pool is non-empty)
        sp = 0.4 if selfplay_pool else 0.0
        bfsp = 1.0 - sp
        return (0.0, 0.0, 0.0, bfsp, sp)


def pick_opponent(episode):
    pp, rp, bp, bfsp, sp = opponent_mix(episode)

    # Zero out selfplay share if pool is empty; redistribute to BFS
    if not selfplay_pool and sp > 0:
        bfsp += sp
        sp = 0.0

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

    # Fallback (floating point edge case)
    return bfs_opp, "bfs"


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------
def get_nearest_item_manhattan(obs):
    raw_map = obs['map_features']['tile_type']
    my_pos  = obs['units']['position'][0]
    item_locs = np.argwhere(raw_map == 2)
    if len(item_locs) == 0:
        return None
    return float(np.abs(item_locs - my_pos).sum(axis=1).min())


def shape_reward(obs, next_obs, raw_reward, distance_shape_weight):
    """
    Reward shaping:
      +3    on item collection (raw_reward > 0)
      -1    extra wall penalty (env already gives -2)
      +/-1.5 * lead_delta  — competitive signal, reduced vs previous
                             version to avoid double-counting collection
      tiny distance hint, decayed to 0 by episode 800
    """
    shaped = float(raw_reward)

    if raw_reward > 0:
        shaped += 3.0
    if raw_reward == -2.0:
        shaped -= 1.0

    my_pts_before  = float(obs['team_points'][0])
    opp_pts_before = float(obs['team_points'][1])
    my_pts_after   = float(next_obs['team_points'][0])
    opp_pts_after  = float(next_obs['team_points'][1])
    lead_delta = (my_pts_after - my_pts_before) - (opp_pts_after - opp_pts_before)
    shaped += 1.5 * lead_delta   # reduced from 2.0 to limit double-counting

    if distance_shape_weight > 0:
        d_before = get_nearest_item_manhattan(obs)
        d_after  = get_nearest_item_manhattan(next_obs)
        if d_before is not None and d_after is not None:
            shaped += distance_shape_weight * (d_before - d_after)

    return shaped


def distance_weight_schedule(episode):
    if episode >= 1500:
        return 0.0
    return 0.2 * (1.0 - episode / 1500.0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_against(agent, opponent, env, n=10):
    """Greedy eval (epsilon=0). Returns (mean_my_score, mean_opp_score, win_rate)."""
    old_eps = agent.epsilon
    agent.epsilon = 0.0
    my_scores, opp_scores, wins = [], [], 0
    for _ in range(n):
        obs, info = env.reset(options=dict(params=EnvParams()))
        episode_max_steps = np.random.randint(MIN_TRAIN_STEPS, MAX_TRAIN_STEPS + 1)
        agent.reset_episode()
        
        done, steps = False, 0
        while not done and steps < episode_max_steps:
            a = agent.act(obs["player_0"])
            o = opponent.act(obs["player_1"])
            obs, _, terminated, truncated, info = env.step({"player_0": a, "player_1": o})
            done = terminated or truncated
            steps += 1
        my  = int(info['state'].team_points[0])
        opp = int(info['state'].team_points[1])
        my_scores.append(my)
        opp_scores.append(opp)
        if my > opp:
            wins += 1
    agent.epsilon = old_eps
    return np.mean(my_scores), np.mean(opp_scores), wins / n


# ---------------------------------------------------------------------------
# Best-checkpoint scoring
# Combines baseline win rate (primary) and BFS score margin (secondary).
# This prevents the checkpoint from plateauing once baseline is consistently
# beaten before BFS training really starts.
# ---------------------------------------------------------------------------
def checkpoint_score(wr_baseline, my_bfs, opp_bfs):
    """Higher is better. Baseline win rate is primary, BFS margin secondary."""
    bfs_margin = (my_bfs - opp_bfs) / max(my_bfs + opp_bfs, 1.0)  # in [-1, 1]
    return wr_baseline + 0.2 * bfs_margin


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
episode_rewards  = []
episode_scores   = []
episode_lengths  = []   # track actual steps per episode
best_ckpt_score  = -999.0

print(f"Starting training. device={agent.device}, "
      f"epsilon={agent.epsilon:.3f}, hidden_dim={agent.hidden_dim}, "
      f"num_episodes={NUM_EPISODES}")
print("Phases: 0-300 passive | 300-800 random | 800-1600 baseline | "
      "1600-3500 BFS | 3500-5000 BFS+selfplay")

for episode in range(NUM_EPISODES):
    opponent, opp_name = pick_opponent(episode)
    dshape_w = distance_weight_schedule(episode)

    obs, info = env.reset(options=dict(params=env_params))
    agent.reset_episode()
    episode_max_steps = np.random.randint(MIN_TRAIN_STEPS, MAX_TRAIN_STEPS + 1)

    total_reward = 0.0
    done  = False
    steps = 0

    while not done and steps < episode_max_steps:
        action     = agent.act(obs["player_0"])
        opp_action = opponent.act(obs["player_1"])
        actions    = {"player_0": action, "player_1": opp_action}

        next_obs, reward, terminated, truncated, info = env.step(actions)

        raw_r    = float(reward[0])
        shaped_r = shape_reward(obs["player_0"], next_obs["player_0"], raw_r, dshape_w)

        done = terminated or truncated
        agent.store(next_obs["player_0"], shaped_r, done)
        if steps % TRAIN_EVERY == 0:
            agent.train_step()

        obs           = next_obs
        total_reward += raw_r
        steps        += 1

    episode_rewards.append(total_reward)
    episode_scores.append(int(info['state'].team_points[0]))
    episode_lengths.append(steps)
    agent.end_episode()

    # ------------------------------------------------------------------
    # Selfplay snapshot
    # ------------------------------------------------------------------
    if (episode + 1) >= SELFPLAY_START_EP and \
       (episode + 1) % SELFPLAY_SNAPSHOT_EVERY == 0 and \
       agent.q_net is not None:
        snap = FrozenSelfOpponent(agent)
        selfplay_pool.append(snap)
        if len(selfplay_pool) > SELFPLAY_POOL_SIZE:
            selfplay_pool.pop(0)
        print(f"  [selfplay] snapshot taken at ep {episode+1} "
              f"(pool size = {len(selfplay_pool)})")
    # ------------------------------------------------------------------
    # Periodic print
    # ------------------------------------------------------------------
    if (episode + 1) % PRINT_EVERY == 0:
        recent_scores   = episode_scores[-PRINT_EVERY:]
        recent_lengths  = episode_lengths[-PRINT_EVERY:]
        avg_s    = np.mean(recent_scores)
        avg_len  = np.mean(recent_lengths)
        # Score per 100 steps: comparable across different episode lengths
        avg_s_per_100 = np.mean([
            s / l * 100 for s, l in zip(recent_scores, recent_lengths)
        ])
        phase = (
            "passive"  if episode < 300  else
            "random"   if episode < 800  else
            "baseline" if episode < 1600 else
            "bfs"      if episode < 3500 else
            "bfs+self"
        )
        print(f"Ep {episode+1:5d} [{phase:8s}] opp={opp_name:8s} | "
              f"avg_score={avg_s:.1f} | score/100steps={avg_s_per_100:.2f} | "
              f"avg_len={avg_len:.0f} | "
              f"eps={agent.epsilon:.3f} | buf={len(agent.replay_buffer)} | "
              f"dshape={dshape_w:.3f}")
        
    # ------------------------------------------------------------------
    # Periodic evaluation
    # ------------------------------------------------------------------
    if (episode + 1) % EVAL_EVERY == 0:
        my_b,  opp_b,  wr_b  = evaluate_against(agent, baseline_opp, env, n=EVAL_GAMES)
        my_r,  opp_r,  wr_r  = evaluate_against(agent, random_opp,   env, n=EVAL_GAMES)
        my_bfs, opp_bfs, wr_bfs = evaluate_against(agent, bfs_opp,   env, n=EVAL_GAMES)

        print(f"  [eval ep {episode+1}] "
              f"vs baseline: {my_b:.1f}-{opp_b:.1f} wr={wr_b:.0%} | "
              f"vs random: {my_r:.1f}-{opp_r:.1f} wr={wr_r:.0%} | "
              f"vs bfs: {my_bfs:.1f}-{opp_bfs:.1f} wr={wr_bfs:.0%}")

        # Save best checkpoint based on combined score
        score = checkpoint_score(wr_b, my_bfs, opp_bfs)
        if score > best_ckpt_score:
            best_ckpt_score = score
            agent.save(filename="weights_best.pth")
            print(f"  [eval] new best checkpoint (score={score:.3f}, "
                  f"baseline_wr={wr_b:.0%}, bfs_margin={my_bfs-opp_bfs:.1f})")


    # ------------------------------------------------------------------
    # Periodic save (latest weights)
    # ------------------------------------------------------------------
    if (episode + 1) % SAVE_EVERY == 0:
        agent.save()
        print(f"  [save] ep {episode+1}: latest weights.pth saved "
              f"(best ckpt score so far: {best_ckpt_score:.3f})")

# Final save
agent.save()
print(f"\nTraining complete after {NUM_EPISODES} episodes.")
print(f"Best checkpoint score: {best_ckpt_score:.3f}")
print("Tip: if weights_best.pth outperforms weights.pth, copy it before submitting.")
env.close()