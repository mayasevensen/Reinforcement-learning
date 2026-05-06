"""
Training script for the CNN-based Collector DQN agent.

Curriculum (revised, 6 phases over 5000 episodes):
  Phase 1 (ep    0– 150): Passive opponent.
  Phase 2 (ep  150– 350): vs Random only.
  Phase 3 (ep  350–1200): vs Random + Baseline (baseline-heavy).
  Phase 4 (ep 1200–2500): vs Baseline + BFS (BFS ramps in slowly).
  Phase 5 (ep 2500–4000): vs Baseline + BFS + Selfplay (selfplay introduced).
  Phase 6 (ep 4000–5000): Tournament-like mix of all three competitive opponents.

Rationale: passive and random phases are short because they teach little
once the agent gets the basics; the baseline phase is extended because
it's the richest source of competitive learning signal; selfplay is
introduced earlier so the snapshot pool diversifies before training ends;
baseline never fully drops out, since most tournament opponents will look
more like baseline than like BFS.

Differences from the previous trainCNN.py:
  - Imports the new (smaller, faster) Agent and preprocess_obs from agent.py.
  - Epsilon decay is now per-step inside the agent itself (driven by
    agent.store()), so we no longer call agent.end_episode() purely for
    decay -- but we still call it so it can reset per-episode state.
  - FrozenSelfOpponent unchanged in spirit but uses the new state dict
    interface ({"grid", "global"}).
  - Checkpoint score weighs the BFS margin a bit more, since BFS is the
    realistic difficulty bar.
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
# Tee print() to log file
# ---------------------------------------------------------------------------
os.makedirs("plots", exist_ok=True)
LOG_PATH = "plots/training_log_cnn.txt"
_log_file = open(LOG_PATH, "w", buffering=1)
_orig_print = builtins.print

def _tee_print(*args, **kwargs):
    _orig_print(*args, **kwargs)
    kwargs.pop("file", None)
    _orig_print(*args, file=_log_file, **kwargs)

builtins.print = _tee_print
import atexit
atexit.register(_log_file.close)
print(f"[log] writing CNN training log to {os.path.abspath(LOG_PATH)}")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
with open("src/agents/agent_cnn/config.yaml") as f:
    config_dict = yaml.safe_load(f)

class Config:
    pass

config = Config()
for k, v in config_dict.items():
    setattr(config, k, v)

config.training = True

sys.path.insert(0, "src/agents/agent_cnn")
from agent import Agent, preprocess_obs  # noqa: E402

agent = Agent(config)
agent.load()  # resume if weights exist

env        = CollectorGymEnv(numpy_output=True)
env_params = EnvParams()


# ---------------------------------------------------------------------------
# Training hyperparameters  (identical to train.py)
# ---------------------------------------------------------------------------
NUM_EPISODES    = 5000
MIN_TRAIN_STEPS = 240
MAX_TRAIN_STEPS = 900
SAVE_EVERY      = 500
PRINT_EVERY     = 50
EVAL_EVERY      = 200
EVAL_GAMES      = 10
TRAIN_EVERY     = getattr(config, 'train_every', 4)

SELFPLAY_POOL_SIZE      = 8
SELFPLAY_SNAPSHOT_EVERY = 250
SELFPLAY_START_EP       = 2500


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

    module_name = f"opp_agent_{os.path.basename(os.path.normpath(agent_dir))}"
    spec = spec_from_file_location(module_name, os.path.join(agent_dir, "agent.py"))
    mod  = module_from_spec(spec)
    spec.loader.exec_module(mod)
    opp  = mod.Agent(cfg)
    opp.load()
    return opp


passive_opp  = PassiveOpponent()
random_opp   = load_opponent("src/agents/random/")
baseline_opp = load_opponent("src/agents/baseline/")
bfs_opp      = load_opponent("src/agents/bfs/")

print("[init] all opponents loaded: passive, random, baseline, bfs")


# ---------------------------------------------------------------------------
# Frozen self-play opponent
# ---------------------------------------------------------------------------
class FrozenSelfOpponent:
    """Frozen snapshot of the agent's q_net used as a self-play opponent."""
    def __init__(self, source_agent):
        import torch
        self.q_net  = copy.deepcopy(source_agent.q_net).to(source_agent.device)
        self.q_net.eval()
        self.device = source_agent.device
        self._torch = torch

    def act(self, obs):
        state = preprocess_obs(obs)
        with self._torch.no_grad():
            grid = self._torch.from_numpy(state["grid"]).unsqueeze(0).to(self.device)
            gvec = self._torch.from_numpy(state["global"]).unsqueeze(0).to(self.device)
            return int(self.q_net(grid, gvec).argmax(dim=1).item())


selfplay_pool = []


# ---------------------------------------------------------------------------
# Curriculum (revised)
#
# Returns probabilities (passive, random, baseline, bfs, selfplay).
# Probabilities always sum to 1.0 within each phase.
#
# Phase 1 (   0– 150) : passive 100%
#     Solo learning. Just enough episodes to lock in "items good, walls bad".
#     Shorter than the original 300; against a stationary opponent the agent
#     stops learning new things quickly.
#
# Phase 2 ( 150– 350) : random 100%
#     Bridge phase. Introduces a moving opponent that doesn't actually
#     compete for items. Kept short -- random opponents can encourage
#     the agent to ignore opponents entirely, which is bad later.
#
# Phase 3 ( 350–1200) : baseline ramping 50% -> 90%, random ramping 50% -> 10%
#     The pedagogically richest phase. Baseline is the first opponent
#     that genuinely competes for items with a coherent strategy, so
#     this is where the agent learns contested item collection.
#
# Phase 4 (1200–2500) : baseline 40%, bfs ramping 30% -> 60%, random fills rest
#     BFS introduced gradually. Baseline kept at meaningful share so
#     the agent doesn't only see optimal opponents -- baseline is closer
#     to the imperfect strategies most tournament opponents will use.
#
# Phase 5 (2500–4000) : baseline 20%, bfs 40%, selfplay 40% (or bfs 80% until pool fills)
#     Selfplay introduced earlier than before so the snapshot pool gets
#     populated and diversified before training ends. ~5 snapshots will
#     be taken during this phase.
#
# Phase 6 (4000–5000) : baseline 30%, bfs 30%, selfplay 40%
#     Tournament conditions: a balanced mix of imperfect (baseline),
#     optimal (bfs), and adaptive (selfplay) opponents.
# ---------------------------------------------------------------------------
def opponent_mix(episode):
    # Phase 1: passive
    if episode < 150:
        return (1.0, 0.0, 0.0, 0.0, 0.0)

    # Phase 2: random only
    if episode < 350:
        return (0.0, 1.0, 0.0, 0.0, 0.0)

    # Phase 3: random -> baseline ramp
    if episode < 1200:
        t  = (episode - 350) / 850.0          # 0 -> 1 across the phase
        rp = max(0.10, 0.50 - 0.40 * t)        # 0.50 -> 0.10
        bp = min(0.90, 0.50 + 0.40 * t)        # 0.50 -> 0.90
        return (0.0, rp, bp, 0.0, 0.0)

    # Phase 4: baseline + BFS (BFS ramps in)
    if episode < 2500:
        t    = (episode - 1200) / 1300.0       # 0 -> 1
        bfsp = min(0.60, 0.30 + 0.30 * t)      # 0.30 -> 0.60
        bp   = 0.40                            # baseline held steady
        rp   = max(0.0, 1.0 - bp - bfsp)        # remainder = random
        return (0.0, rp, bp, bfsp, 0.0)

    # Phase 5: introduce selfplay (or extra BFS until pool fills)
    if episode < 4000:
        if selfplay_pool:
            return (0.0, 0.0, 0.20, 0.40, 0.40)
        else:
            # No snapshots yet -- compensate with extra BFS.
            return (0.0, 0.0, 0.20, 0.80, 0.0)

    # Phase 6: tournament-like mix
    if selfplay_pool:
        return (0.0, 0.0, 0.30, 0.30, 0.40)
    else:
        return (0.0, 0.0, 0.30, 0.70, 0.0)


def pick_opponent(episode):
    pp, rp, bp, bfsp, sp = opponent_mix(episode)
    if not selfplay_pool and sp > 0:
        bfsp += sp
        sp = 0.0

    r, cumulative = np.random.random(), 0.0
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
# Reward shaping  (identical to train.py)
# ---------------------------------------------------------------------------
def get_nearest_item_manhattan(obs):
    raw_map  = obs['map_features']['tile_type']
    my_pos   = obs['units']['position'][0]
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
    lead_delta = (my_pts_after - my_pts_before) - (opp_pts_after - opp_pts_before)
    shaped += 1.5 * lead_delta

    if distance_shape_weight > 0:
        d_before = get_nearest_item_manhattan(obs)
        d_after  = get_nearest_item_manhattan(next_obs)
        if d_before is not None and d_after is not None:
            shaped += distance_shape_weight * (d_before - d_after)

    return shaped


def distance_weight_schedule(episode):
    # Decay distance shaping over phases 1+2 (passive + random).
    # Once baseline appears (ep 350+), "get closer to the nearest item"
    # can mislead the agent: it may want to head AWAY from items the
    # opponent will reach first. Turn the bonus off before that happens.
    if episode >= 350:
        return 0.0
    return 0.2 * (1.0 - episode / 350.0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_against(agent, opponent, env, n=10):
    old_eps      = agent.epsilon
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
            done  = terminated or truncated
            steps += 1

        my  = int(info['state'].team_points[0])
        opp = int(info['state'].team_points[1])
        my_scores.append(my)
        opp_scores.append(opp)
        if my > opp:
            wins += 1

    agent.epsilon = old_eps
    return np.mean(my_scores), np.mean(opp_scores), wins / n


def checkpoint_score(wr_baseline, my_bfs, opp_bfs):
    """Slightly higher weight on BFS margin -- it tracks competitive skill better."""
    bfs_margin = (my_bfs - opp_bfs) / max(my_bfs + opp_bfs, 1.0)
    return wr_baseline + 0.5 * bfs_margin


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
episode_rewards = []
episode_scores  = []
episode_lengths = []
best_ckpt_score = -999.0

print(f"Starting CNN training. device={agent.device}, "
      f"epsilon={agent.epsilon:.3f}, hidden_dim={agent.hidden_dim}, "
      f"num_episodes={NUM_EPISODES}")
print("Phases: 0-150 passive | 150-350 random | 350-1200 random+baseline | "
      "1200-2500 baseline+BFS | 2500-4000 +selfplay | 4000-5000 tournament-mix")

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

    # --- Selfplay snapshot ---
    if (episode + 1) >= SELFPLAY_START_EP and \
       (episode + 1) % SELFPLAY_SNAPSHOT_EVERY == 0 and \
       agent.q_net is not None:
        snap = FrozenSelfOpponent(agent)
        selfplay_pool.append(snap)
        if len(selfplay_pool) > SELFPLAY_POOL_SIZE:
            selfplay_pool.pop(0)
        print(f"  [selfplay] snapshot taken at ep {episode+1} "
              f"(pool size = {len(selfplay_pool)})")

    # --- Periodic print ---
    if (episode + 1) % PRINT_EVERY == 0:
        recent_scores  = episode_scores[-PRINT_EVERY:]
        recent_lengths = episode_lengths[-PRINT_EVERY:]
        avg_s     = np.mean(recent_scores)
        avg_len   = np.mean(recent_lengths)
        avg_s_per_100 = np.mean([
            s / l * 100 for s, l in zip(recent_scores, recent_lengths)
        ])
        phase = (
            "passive"  if episode < 150  else
            "random"   if episode < 350  else
            "baseline" if episode < 1200 else
            "bfs"      if episode < 2500 else
            "+self"    if episode < 4000 else
            "tourney"
        )
        print(f"Ep {episode+1:5d} [{phase:8s}] opp={opp_name:8s} | "
              f"avg_score={avg_s:.1f} | score/100steps={avg_s_per_100:.2f} | "
              f"avg_len={avg_len:.0f} | "
              f"eps={agent.epsilon:.3f} | buf={len(agent.replay_buffer) if agent.replay_buffer else 0} | "
              f"dshape={dshape_w:.3f}")

    # --- Periodic eval ---
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
            agent.save(filename="weights_best.pth")
            print(f"  [eval] new best checkpoint (score={score:.3f}, "
                  f"baseline_wr={wr_b:.0%}, bfs_margin={my_bfs-opp_bfs:.1f})")

    # --- Periodic save ---
    if (episode + 1) % SAVE_EVERY == 0:
        agent.save()
        print(f"  [save] ep {episode+1}: weights.pth saved "
              f"(best ckpt score so far: {best_ckpt_score:.3f})")

agent.save()
print(f"\nCNN training complete after {NUM_EPISODES} episodes.")
print(f"Best checkpoint score: {best_ckpt_score:.3f}")
print("Tip: if weights_best.pth outperforms weights.pth, copy it before submitting.")
env.close()