"""
Training script for the CNN-based Collector DQN agent.

Trains the agent through a 6-phase opponent curriculum, periodically
evaluating against the rule-based opponents (random, baseline, BFS) and
saving the best checkpoint by a combined baseline-winrate / BFS-margin
score. From phase 5 onward, frozen snapshots of the agent are added to
a self-play pool that is sampled alongside the rule-based opponents.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import atexit
import builtins
import copy
from importlib.util import spec_from_file_location, module_from_spec

import numpy as np
import yaml

from environments.collector.params import EnvParams
from environments.collector.wrappers import CollectorGymEnv


# ---------------------------------------------------------------------------
# Reward shaping constants
# ---------------------------------------------------------------------------
# Total shaped reward per event. These OVERWRITE the env's raw reward
# (which is +1 collect, -2 wall, -1 step). Tune these to retune how
# strongly the agent feels each event before competitive/distance shaping
# is added on top.
HIT_WALL = -5.0
STEP = -1.0
ITEM = 5.0

# Multiplier on (my_score_delta - opp_score_delta). Gives the agent a
# direct gradient toward beating the opponent rather than just scoring.
LEAD_DELTA_WEIGHT = 1.5

# Distance-to-nearest-item shaping: small bonus for moving closer to an
# item. Used as training wheels in the early curriculum phases and decays
# linearly to zero by DSHAPE_OFF_EP, so it's gone before competitive
# opponents arrive.
DSHAPE_INITIAL = 0.2
DSHAPE_OFF_EP = 260


# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
NUM_EPISODES = 3000
MIN_TRAIN_STEPS = 200  # episode-length cap is randomised per episode
MAX_TRAIN_STEPS = 1000  # to discourage policies that depend on horizon
SAVE_EVERY = 500
PRINT_EVERY = 50
EVAL_EVERY = 200
EVAL_GAMES = 30

# Self-play snapshots: from SELFPLAY_START_EP onward, every
# SELFPLAY_SNAPSHOT_EVERY episodes we freeze a copy of the q-net and add
# it to the pool. Pool keeps the SELFPLAY_POOL_SIZE most recent snapshots.
SELFPLAY_POOL_SIZE = 8
SELFPLAY_SNAPSHOT_EVERY = 150
SELFPLAY_START_EP = 1500

# Curriculum phase boundaries (in episodes). Used by both opponent_mix()
# and the phase-name lookup in the periodic print. Keeping them as named
# constants makes it impossible for the two to drift out of sync.
PHASE_1_END = 90  # passive only
PHASE_2_END = 210  # random only
PHASE_3_END = 720  # random -> baseline ramp
PHASE_4_END = 1500  # baseline + BFS ramp
PHASE_5_END = 2400  # baseline + BFS + self-play introduced
# episodes after PHASE_5_END are tournament-mix


# ---------------------------------------------------------------------------
# Logging: tee print() to a log file
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
atexit.register(_log_file.close)
print(f"[log] writing CNN training log to {os.path.abspath(LOG_PATH)}")


# ---------------------------------------------------------------------------
# Config + agent + environment
# ---------------------------------------------------------------------------
# Force training=True regardless of what the yaml says — the same config
# file is also used at evaluation time where it's set to False.
with open("src/agents/agent_cnn/config.yaml") as f:
    config_dict = yaml.safe_load(f)

class Config:
    pass

config = Config()
for k, v in config_dict.items():
    setattr(config, k, v)
config.training = True

sys.path.insert(0, "src/agents/agent_cnn")
from agent import Agent, FrameStacker, preprocess_obs  # noqa: E402

agent = Agent(config)
agent.load()  # resume from disk if weights exist; silently no-ops otherwise

env = CollectorGymEnv(numpy_output=True)
env_params = EnvParams()

TRAIN_EVERY = getattr(config, 'train_every', 1)


# ---------------------------------------------------------------------------
# Opponents
# ---------------------------------------------------------------------------
class PassiveOpponent:
    """Always returns action 0 — used in the easiest curriculum phase."""
    def act(self, obs):
        return 0


def load_opponent(agent_dir):
    """
    Load an external Agent (random / baseline / BFS) from its directory.
    Uses importlib so each opponent's `Agent` class is loaded as its own
    uniquely-named module — otherwise multiple `from agent import Agent`
    statements would clash.
    """
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


passive_opp = PassiveOpponent()
random_opp = load_opponent("src/agents/random/")
baseline_opp = load_opponent("src/agents/baseline/")
bfs_opp = load_opponent("src/agents/bfs/")
print("[init] opponents loaded: passive, random, baseline, bfs")


class FrozenSelfOpponent:
    """
    Frozen snapshot of the agent's q-net used as a self-play opponent.

    Carries its own FrameStacker because it sees the game from player_1's
    (flipped) perspective and needs an independent frame history. Must be
    reset between episodes via reset_episode().
    """
    def __init__(self, source_agent):
        import torch
        self.q_net = copy.deepcopy(source_agent.q_net).to(source_agent.device)
        self.q_net.eval()
        self.device = source_agent.device
        self._torch = torch
        self._stacker = FrameStacker()

    def reset_episode(self):
        self._stacker.reset()

    def act(self, obs):
        state = self._stacker.step(obs)
        with self._torch.no_grad():
            grid = self._torch.from_numpy(state["grid"]).unsqueeze(0).to(self.device)
            gvec = self._torch.from_numpy(state["global"]).unsqueeze(0).to(self.device)
            return int(self.q_net(grid, gvec).argmax(dim=1).item())


selfplay_pool = []  # populated from PHASE_5_START onward


def maybe_reset_opponent(opponent):
    """Self-play opponents have a frame-history to reset; rule-based ones don't."""
    reset_fn = getattr(opponent, 'reset_episode', None)
    if callable(reset_fn):
        reset_fn()


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------
# opponent_mix() returns probabilities (passive, random, baseline, bfs,
# selfplay) summing to 1.0. Rationale per phase:
#   1. Passive: lock in basic navigation and "items good, walls bad"
#      without distractions.
#   2. Random: introduce a moving opponent that doesn't compete, as a
#      bridge to genuine competition.
#   3. Random→baseline ramp: baseline is the first opponent that actually
#      competes for items, and most tournament agents resemble it more
#      than they resemble BFS — this is the richest learning phase.
#   4. Baseline + BFS ramp: introduce optimal pathfinding gradually while
#      keeping baseline at a meaningful share.
#   5. Add self-play: build up a snapshot pool while baseline and BFS
#      keep providing diverse signal.
#   6. Tournament-like mix: balanced exposure to imperfect (baseline),
#      optimal (bfs), and adaptive (self-play) opponents.
def opponent_mix(episode):
    if episode < PHASE_1_END:
        return (1.0, 0.0, 0.0, 0.0, 0.0)

    if episode < PHASE_2_END:
        return (0.0, 1.0, 0.0, 0.0, 0.0)

    if episode < PHASE_3_END:
        # Random 50%→10%, baseline 50%→90% over the phase.
        t = (episode - PHASE_2_END) / (PHASE_3_END - PHASE_2_END)
        rp = max(0.10, 0.50 - 0.40 * t)
        bp = min(0.90, 0.50 + 0.40 * t)
        return (0.0, rp, bp, 0.0, 0.0)

    if episode < PHASE_4_END:
        # BFS ramps 30%→60%, baseline fixed at 40%, random fills the rest.
        t = (episode - PHASE_3_END) / (PHASE_4_END - PHASE_3_END)
        bfsp = min(0.60, 0.30 + 0.30 * t)
        bp = 0.40
        rp = max(0.0, 1.0 - bp - bfsp)
        return (0.0, rp, bp, bfsp, 0.0)

    if episode < PHASE_5_END:
        # Selfplay 40% once the pool has any snapshots, otherwise route
        # that share to BFS so we never sample from an empty pool.
        if selfplay_pool:
            return (0.0, 0.0, 0.20, 0.40, 0.40)
        return (0.0, 0.0, 0.20, 0.80, 0.0)

    # Phase 6 (final): tournament-like mix.
    if selfplay_pool:
        return (0.0, 0.0, 0.30, 0.30, 0.40)
    return (0.0, 0.0, 0.30, 0.70, 0.0)


def pick_opponent(episode):
    """Sample one opponent from this episode's mixture."""
    pp, rp, bp, bfsp, sp = opponent_mix(episode)
    # Belt-and-braces: redirect any selfplay probability to BFS if the
    # pool is empty. opponent_mix() already handles this, but doing it
    # here too means an empty pool can never crash np.random.choice.
    if not selfplay_pool and sp > 0:
        bfsp += sp
        sp = 0.0

    r, cumulative = np.random.random(), 0.0
    for prob, opp, name in [
        (pp, passive_opp, "passive"),
        (rp, random_opp, "random"),
        (bp, baseline_opp, "baseline"),
        (bfsp, bfs_opp, "bfs"),
        (sp, None, "selfplay"),
    ]:
        cumulative += prob
        if r < cumulative:
            if name == "selfplay":
                return np.random.choice(selfplay_pool), "selfplay"
            return opp, name

    # Floating-point fallback if the probabilities don't quite sum to 1.
    return bfs_opp, "bfs"


def phase_name(episode):
    """Short label for log lines."""
    if episode < PHASE_1_END: return "passive"
    if episode < PHASE_2_END: return "random"
    if episode < PHASE_3_END: return "baseline"
    if episode < PHASE_4_END: return "bfs"
    if episode < PHASE_5_END: return "+self"
    return "tourney"


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------
def get_nearest_item_manhattan(obs):
    raw_map = obs['map_features']['tile_type']
    my_pos = obs['units']['position'][0]
    item_locs = np.argwhere(raw_map == 2)
    if len(item_locs) == 0:
        return None
    return float(np.abs(item_locs - my_pos).sum(axis=1).min())


def shape_reward(obs, next_obs, raw_reward, distance_shape_weight):
    # Map the env's raw event reward to the configured total.
    if raw_reward > 0:
        shaped = ITEM
    elif raw_reward == -2.0:
        shaped = HIT_WALL
    else:
        shaped = STEP

    # Did my lead over the opponent improve this turn?
    my_pts_before = float(obs['team_points'][0])
    opp_pts_before = float(obs['team_points'][1])
    my_pts_after = float(next_obs['team_points'][0])
    opp_pts_after = float(next_obs['team_points'][1])
    lead_delta = (my_pts_after - my_pts_before) - (opp_pts_after - opp_pts_before)
    shaped += LEAD_DELTA_WEIGHT * lead_delta

    # Optional distance-to-item bonus (weight is zero after early phases).
    if distance_shape_weight > 0:
        d_before = get_nearest_item_manhattan(obs)
        d_after = get_nearest_item_manhattan(next_obs)
        if d_before is not None and d_after is not None:
            shaped += distance_shape_weight * (d_before - d_after)

    return shaped


def distance_weight_schedule(episode):
    """Linear decay from DSHAPE_INITIAL at ep 0 to 0 at ep DSHAPE_OFF_EP."""
    if episode >= DSHAPE_OFF_EP:
        return 0.0
    return DSHAPE_INITIAL * (1.0 - episode / DSHAPE_OFF_EP)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_against(agent, opponent, env, n=10):
    """Greedy (epsilon=0) eval over n episodes. Returns (my_avg, opp_avg, win_rate)."""
    old_eps = agent.epsilon
    agent.epsilon = 0.0
    my_scores, opp_scores, wins = [], [], 0

    for _ in range(n):
        obs, info = env.reset(options=dict(params=EnvParams()))
        episode_max_steps = np.random.randint(MIN_TRAIN_STEPS, MAX_TRAIN_STEPS + 1)
        agent.reset_episode()
        maybe_reset_opponent(opponent)
        done, steps = False, 0

        while not done and steps < episode_max_steps:
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


def checkpoint_score(wr_baseline, my_bfs, opp_bfs):
    """
    Single scalar for ranking checkpoints. Baseline win rate is the primary
    signal (most tournament opponents resemble baseline); a normalised BFS
    margin acts as a tiebreaker, since BFS is the harder skill ceiling.
    """
    bfs_margin = (my_bfs - opp_bfs) / max(my_bfs + opp_bfs, 1.0)
    return wr_baseline + 0.5 * bfs_margin


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
episode_rewards = []
episode_scores = []
episode_lengths = []
best_ckpt_score = -999.0

print(f"Starting CNN training. device={agent.device}, "
      f"epsilon={agent.epsilon:.3f}, hidden_dim={agent.hidden_dim}, "
      f"num_episodes={NUM_EPISODES}")
print(f"Phases: 0-{PHASE_1_END} passive | "
      f"{PHASE_1_END}-{PHASE_2_END} random | "
      f"{PHASE_2_END}-{PHASE_3_END} random+baseline | "
      f"{PHASE_3_END}-{PHASE_4_END} baseline+BFS | "
      f"{PHASE_4_END}-{PHASE_5_END} +selfplay | "
      f"{PHASE_5_END}-{NUM_EPISODES} tournament-mix")

for episode in range(NUM_EPISODES):
    opponent, opp_name = pick_opponent(episode)
    dshape_w = distance_weight_schedule(episode)

    obs, info = env.reset(options=dict(params=env_params))
    agent.reset_episode()
    maybe_reset_opponent(opponent)
    episode_max_steps = np.random.randint(MIN_TRAIN_STEPS, MAX_TRAIN_STEPS + 1)

    total_reward, done, steps = 0.0, False, 0

    while not done and steps < episode_max_steps:
        action = agent.act(obs["player_0"])
        opp_action = opponent.act(obs["player_1"])
        next_obs, reward, terminated, truncated, info = env.step(
            {"player_0": action, "player_1": opp_action}
        )

        raw_r = float(reward[0])
        shaped_r = shape_reward(obs["player_0"], next_obs["player_0"], raw_r, dshape_w)

        done = terminated or truncated
        agent.store(next_obs["player_0"], shaped_r, done)
        if steps % TRAIN_EVERY == 0:
            agent.train_step()

        obs = next_obs
        total_reward += raw_r  # tracked from raw reward for honest reporting
        steps += 1

    episode_rewards.append(total_reward)
    episode_scores.append(int(info['state'].team_points[0]))
    episode_lengths.append(steps)
    agent.end_episode()

    # ---- Self-play snapshot ------------------------------------------------
    if (episode + 1) >= SELFPLAY_START_EP and \
       (episode + 1) % SELFPLAY_SNAPSHOT_EVERY == 0 and \
       agent.q_net is not None:
        selfplay_pool.append(FrozenSelfOpponent(agent))
        if len(selfplay_pool) > SELFPLAY_POOL_SIZE:
            selfplay_pool.pop(0)
        print(f"  [selfplay] snapshot taken at ep {episode+1} "
              f"(pool size = {len(selfplay_pool)})")

    # ---- Periodic status print --------------------------------------------
    if (episode + 1) % PRINT_EVERY == 0:
        recent_scores = episode_scores[-PRINT_EVERY:]
        recent_lengths = episode_lengths[-PRINT_EVERY:]
        avg_s = np.mean(recent_scores)
        avg_len = np.mean(recent_lengths)
        # Score-per-100-steps is more meaningful than raw average score
        # because episode lengths are randomised.
        avg_s_per_100 = np.mean([s / l * 100 for s, l in zip(recent_scores, recent_lengths)])
        buf_size = len(agent.replay_buffer) if agent.replay_buffer else 0
        print(f"Ep {episode+1:5d} [{phase_name(episode):8s}] opp={opp_name:8s} | "
              f"avg_score={avg_s:.1f} | score/100steps={avg_s_per_100:.2f} | "
              f"avg_len={avg_len:.0f} | "
              f"eps={agent.epsilon:.3f} | buf={buf_size} | dshape={dshape_w:.3f}")

    # ---- Periodic eval + best-checkpoint save -----------------------------
    if (episode + 1) % EVAL_EVERY == 0:
        my_b, opp_b, wr_b = evaluate_against(agent, baseline_opp, env, n=EVAL_GAMES)
        my_r, opp_r, wr_r = evaluate_against(agent, random_opp, env, n=EVAL_GAMES)
        my_bfs, opp_bfs, wr_bfs = evaluate_against(agent, bfs_opp, env, n=EVAL_GAMES)

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

    # ---- Periodic latest-weights save -------------------------------------
    if (episode + 1) % SAVE_EVERY == 0:
        agent.save()
        print(f"  [save] ep {episode+1}: weights.pth saved "
              f"(best ckpt score so far: {best_ckpt_score:.3f})")

agent.save()
print(f"\nCNN training complete after {NUM_EPISODES} episodes.")
print(f"Best checkpoint score: {best_ckpt_score:.3f}")
env.close()