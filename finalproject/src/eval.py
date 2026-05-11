"""
eval.py -- compare weight files for the CNN agent.

Usage examples:

  # Evaluate a single agent dir against all opponents (10 games each):
  python eval.py src/agents/agent_cnn/

  # Compare two weight files head-to-head:
  python eval.py src/agents/agent_cnn/ --weights weights.pth weights_best.pth

  # More games for a more reliable estimate:
  python eval.py src/agents/agent_cnn/ --weights weights.pth weights_best.pth --n 30

  # Also run the two weight files against each other:
  python eval.py src/agents/agent_cnn/ --weights weights.pth weights_best.pth --h2h

The script prints a table like:

  weights.pth
    vs random   :  win=100%  my=81.3  opp=5.9  margin=+75.4
    vs baseline :  win= 90%  my=72.1  opp=25.3  margin=+46.8
    vs bfs      :  win= 70%  my=63.4  opp=55.1  margin=+8.3
    COMPOSITE SCORE: 0.982

  weights_best.pth
    vs random   :  win=100%  my=84.2  opp=6.1  margin=+78.1
    vs baseline :  win=100%  my=78.3  opp=24.1  margin=+54.2
    vs bfs      :  win= 80%  my=67.8  opp=57.2  margin=+10.6
    COMPOSITE SCORE: 1.041

  WINNER: weights_best.pth  (score 1.041 vs 1.032)
"""

import os
import sys
import argparse
import warnings
import importlib.util

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import numpy as np
import torch

from environments.collector.wrappers import CollectorGymEnv
from environments.collector.params import EnvParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_agent_from_dir(agent_dir, weights_filename="weights.pth", training=False):
    """Load an Agent from agent_dir, optionally overriding the weights file."""
    config_path = os.path.join(agent_dir, "config.yaml")
    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)

    class Config:
        pass

    cfg = Config()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    cfg.training = training

    spec = importlib.util.spec_from_file_location(
        "agent_eval", os.path.join(agent_dir, "agent.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    agent = mod.Agent(cfg)

    # Override weights filename if requested
    weights_path = os.path.join(agent_dir, cfg_dict.get("weights_dir", "weights"), weights_filename)
    if os.path.exists(weights_path):
        agent._pending_load_path = weights_path
        print(f"  queued: {weights_path}")
    else:
        print(f"  WARNING: weights not found at {weights_path} -- will start fresh")

    return agent, mod


def load_opponent(agent_dir):
    config_path = os.path.join(agent_dir, "config.yaml")
    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)

    class Cfg:
        pass

    cfg = Cfg()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)

    spec = importlib.util.spec_from_file_location(
        f"opp_{os.path.basename(agent_dir.rstrip('/\\'))}",
        os.path.join(agent_dir, "agent.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    opp = mod.Agent(cfg)
    opp.load()
    return opp


# ---------------------------------------------------------------------------
# Single evaluation run
# ---------------------------------------------------------------------------
def run_games(agent, opponent, env, n=10, max_steps=800, opp_name=""):
    """
    Run n games of agent (player_0) vs opponent (player_1).
    Returns dict with win_rate, avg_my_score, avg_opp_score, avg_margin.
    """
    my_scores, opp_scores, wins = [], [], 0

    # Disable exploration
    old_eps = getattr(agent, "epsilon", 0.0)
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0

    has_reset = hasattr(agent, "reset_episode")
    opp_has_reset = hasattr(opponent, "reset_episode")

    for g in range(n):
        obs, info = env.reset(options=dict(params=EnvParams()))
        if has_reset:
            agent.reset_episode()
        if opp_has_reset:
            opponent.reset_episode()

        done, steps = False, 0
        while not done and steps < max_steps:
            a0 = agent.act(obs["player_0"])
            a1 = opponent.act(obs["player_1"])
            obs, _, terminated, truncated, info = env.step({"player_0": a0, "player_1": a1})
            done = terminated or truncated
            steps += 1

        my  = int(info["state"].team_points[0])
        opp = int(info["state"].team_points[1])
        my_scores.append(my)
        opp_scores.append(opp)
        if my > opp:
            wins += 1

        if (g + 1) % 10 == 0:
            print(f"    {opp_name} game {g+1}/{n}: {my}-{opp}".ljust(50), end="\r")

    if hasattr(agent, "epsilon"):
        agent.epsilon = old_eps

    return {
        "win_rate":   wins / n,
        "my_score":   np.mean(my_scores),
        "opp_score":  np.mean(opp_scores),
        "margin":     np.mean(my_scores) - np.mean(opp_scores),
        "wins":       wins,
        "n":          n,
    }


def composite_score(results):
    """
    Weighted composite: baseline win rate counts most (tournament-relevant),
    BFS margin is a secondary signal. Random is a sanity check.
    """
    wr_b   = results["baseline"]["win_rate"]
    wr_r   = results["random"]["win_rate"]
    margin_bfs = results["bfs"]["margin"]
    total  = results["baseline"]["my_score"] + results["bfs"]["my_score"]
    bfs_norm = margin_bfs / max(total, 1.0)
    return wr_b + 0.1 * wr_r + 0.2 * bfs_norm


def print_results(label, results):
    score = composite_score(results)
    print(f"\n  {label}")
    for opp_name, r in results.items():
        sign = "+" if r["margin"] >= 0 else ""
        print(
            f"    vs {opp_name:<10}: "
            f"win={r['win_rate']:>4.0%}  "
            f"my={r['my_score']:>6.1f}  "
            f"opp={r['opp_score']:>6.1f}  "
            f"margin={sign}{r['margin']:.1f}"
        )
    print(f"    COMPOSITE SCORE: {score:.3f}")
    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN agent weight files.")
    parser.add_argument("agent_dir", help="Path to the agent directory (contains agent.py, config.yaml, weights/)")
    parser.add_argument(
        "--weights", nargs="+", default=["weights_best.pth", "weights.pth"],
        help="Weight filenames to compare (looked up inside weights_dir). Default: weights_best.pth weights.pth"
    )
    parser.add_argument("--n", type=int, default=200, help="Games per opponent per weight file (default 50)")
    parser.add_argument("--h2h", action="store_true", help="Also run the weight files head-to-head against each other")
    parser.add_argument(
        "--opponents", nargs="+",
        default=["src/agents/random/", "src/agents/baseline/", "src/agents/bfs/"],
        help="Opponent agent directories to evaluate against"
    )
    args = parser.parse_args()

    env = CollectorGymEnv(numpy_output=True)

    # Load opponents
    print("\nLoading opponents...")
    opponents = {}
    opp_labels = {
        "src/agents/random/":   "random",
        "src/agents/baseline/": "baseline",
        "src/agents/bfs/":      "bfs",
    }
    for opp_dir in args.opponents:
        label = opp_labels.get(opp_dir.rstrip("/\\").replace("\\", "/"),
                               os.path.basename(opp_dir.rstrip("/\\")))
        try:
            opponents[label] = load_opponent(opp_dir)
            print(f"  loaded: {label} from {opp_dir}")
        except Exception as e:
            print(f"  SKIP {opp_dir}: {e}")

    if not opponents:
        print("No opponents loaded -- check paths.")
        sys.exit(1)

    # Filter to weight files that actually exist
    weights_dir_name = None
    try:
        with open(os.path.join(args.agent_dir, "config.yaml")) as f:
            weights_dir_name = yaml.safe_load(f).get("weights_dir", "weights")
    except Exception:
        weights_dir_name = "weights"

    available = []
    for wf in args.weights:
        full = os.path.join(args.agent_dir, weights_dir_name, wf)
        if os.path.exists(full):
            available.append(wf)
        else:
            print(f"  SKIP {wf}: not found at {full}")

    if not available:
        print("No weight files found.")
        sys.exit(1)

    # Evaluate each weight file
    print(f"\nEvaluating {len(available)} weight file(s) over {args.n} games each...\n")
    all_scores = {}
    all_results = {}

    for wf in available:
        print(f"Loading {wf}...")
        agent, _ = load_agent_from_dir(args.agent_dir, weights_filename=wf)

        results = {}
        for opp_label, opp in opponents.items():
            print(f"  vs {opp_label}...")
            results[opp_label] = run_games(agent, opp, env, n=args.n, opp_name=opp_label)
            print()  # clear \r line

        all_results[wf] = results
        all_scores[wf]  = print_results(wf, results)

    # Head-to-head between weight files
    if args.h2h and len(available) >= 2:
        print("\n" + "="*60)
        print("HEAD-TO-HEAD")
        print("="*60)
        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                wf_a, wf_b = available[i], available[j]
                print(f"\n  {wf_a}  vs  {wf_b}  ({args.n} games)")
                agent_a, _ = load_agent_from_dir(args.agent_dir, weights_filename=wf_a)
                agent_b, _ = load_agent_from_dir(args.agent_dir, weights_filename=wf_b)
                r = run_games(agent_a, agent_b, env, n=args.n, opp_name=wf_b)
                print()
                sign = "+" if r["margin"] >= 0 else ""
                print(
                    f"  {wf_a}: win={r['win_rate']:.0%}  "
                    f"score={r['my_score']:.1f}-{r['opp_score']:.1f}  "
                    f"margin={sign}{r['margin']:.1f}"
                )

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    ranked = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (wf, score) in enumerate(ranked, 1):
        marker = " ← BEST" if rank == 1 else ""
        print(f"  {rank}. {wf:<30} score={score:.3f}{marker}")

    if len(ranked) > 1:
        best, second = ranked[0], ranked[1]
        print(f"\n  WINNER: {best[0]}  (score {best[1]:.3f} vs {second[1]:.3f})")

    env.close()


if __name__ == "__main__":
    main()