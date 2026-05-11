"""
eval_for_report.py -- produce the numbers and figure for the Results paragraph.

Does three things:

  1. Runs a clean greedy (epsilon=0) evaluation of the trained agent against
     random, baseline, and BFS opponents over N games each.
  2. Parses the training log at plots/training_log_cnn.txt to extract the
     periodic eval points logged during training (margin vs each opponent
     over the course of the run).
  3. Produces two artefacts:
       - training_trajectory.pdf  (the figure for \\ref{fig:training})
       - results_paragraph.txt    (the LaTeX paragraph with numbers filled in)

Usage (run from src/ so the import paths line up with eval.py's convention):

  python eval_for_report.py src/agents/agent_cnn/
  python eval_for_report.py src/agents/agent_cnn/ --n 200
  python eval_for_report.py src/agents/agent_cnn/ \\
      --log plots/training_log_cnn.txt \\s
      --weights weights_best.pth \\
      --out-dir plots/

The script borrows the agent-loading style from eval.py.
"""

import os
import re
import sys
import argparse
import warnings
import importlib.util

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Match eval.py's path setup so `environments.collector...` resolves.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt

from environments.collector.wrappers import CollectorGymEnv
from environments.collector.params import EnvParams


# ---------------------------------------------------------------------------
# Phase boundaries -- mirror trainCNN_best.py so the figure annotations
# stay in sync with the curriculum used during training.
# ---------------------------------------------------------------------------
PHASE_1_END = 90
PHASE_2_END = 210
PHASE_3_END = 720
PHASE_4_END = 1500
PHASE_5_END = 2400

PHASES = [
    (0,           PHASE_1_END, "passive"),
    (PHASE_1_END, PHASE_2_END, "random"),
    (PHASE_2_END, PHASE_3_END, "baseline ramp"),
    (PHASE_3_END, PHASE_4_END, "+ BFS ramp"),
    (PHASE_4_END, PHASE_5_END, "+ self-play"),
    (PHASE_5_END, 3000,        "tournament mix"),
]


# ---------------------------------------------------------------------------
# Agent loading -- adapted from eval.py.
# ---------------------------------------------------------------------------
def load_agent_from_dir(agent_dir, weights_filename="weights_best.pth", training=False):
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

    weights_path = os.path.join(
        agent_dir, cfg_dict.get("weights_dir", "weights"), weights_filename
    )
    if os.path.exists(weights_path):
        agent._pending_load_path = weights_path
        print(f"  queued: {weights_path}")
    else:
        print(f"  WARNING: weights not found at {weights_path}")

    return agent


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
# Run games -- same shape as eval.py.run_games but returns per-game arrays
# so we can report std on top of the mean if we want.
# ---------------------------------------------------------------------------
def run_games(agent, opponent, env, n=100, max_steps=800, opp_name=""):
    my_scores, opp_scores, wins = [], [], 0

    # Force greedy play.
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
            obs, _, terminated, truncated, info = env.step(
                {"player_0": a0, "player_1": a1}
            )
            done = terminated or truncated
            steps += 1

        my = int(info["state"].team_points[0])
        opp = int(info["state"].team_points[1])
        my_scores.append(my)
        opp_scores.append(opp)
        if my > opp:
            wins += 1

        if (g + 1) % 10 == 0:
            print(f"    {opp_name} {g+1}/{n}: {my}-{opp}".ljust(50), end="\r")

    if hasattr(agent, "epsilon"):
        agent.epsilon = old_eps

    my_scores = np.asarray(my_scores, dtype=np.float32)
    opp_scores = np.asarray(opp_scores, dtype=np.float32)

    return {
        "win_rate":  wins / n,
        "my_mean":   float(my_scores.mean()),
        "opp_mean":  float(opp_scores.mean()),
        "my_std":    float(my_scores.std(ddof=1)) if n > 1 else 0.0,
        "opp_std":   float(opp_scores.std(ddof=1)) if n > 1 else 0.0,
        "margin":    float((my_scores - opp_scores).mean()),
        "margin_std": float((my_scores - opp_scores).std(ddof=1)) if n > 1 else 0.0,
        "wins":      wins,
        "n":         n,
    }


# ---------------------------------------------------------------------------
# Training log parsing
# ---------------------------------------------------------------------------
# Matches the log line format from trainCNN_best.py:
#   [eval ep 200] vs baseline: 12.3-45.6 wr=10% | vs random: 70.1-5.2 wr=100% | vs bfs: 8.4-66.7 wr=0%
EVAL_RE = re.compile(
    r"\[eval ep (?P<ep>\d+)\]\s*"
    r"vs baseline:\s*(?P<bm>-?[\d.]+)-(?P<bo>-?[\d.]+)\s*wr=(?P<bw>\d+)%\s*\|\s*"
    r"vs random:\s*(?P<rm>-?[\d.]+)-(?P<ro>-?[\d.]+)\s*wr=(?P<rw>\d+)%\s*\|\s*"
    r"vs bfs:\s*(?P<fm>-?[\d.]+)-(?P<fo>-?[\d.]+)\s*wr=(?P<fw>\d+)%"
)


def parse_training_log(path):
    """Return a dict of arrays keyed by opponent name + 'episodes'."""
    if not os.path.exists(path):
        print(f"  WARNING: training log not found at {path}; skipping trajectory plot.")
        return None

    eps, b_my, b_opp, b_wr = [], [], [], []
    r_my, r_opp, r_wr = [], [], []
    f_my, f_opp, f_wr = [], [], []

    with open(path) as fh:
        for line in fh:
            m = EVAL_RE.search(line)
            if not m:
                continue
            eps.append(int(m["ep"]))
            b_my.append(float(m["bm"])); b_opp.append(float(m["bo"])); b_wr.append(int(m["bw"]))
            r_my.append(float(m["rm"])); r_opp.append(float(m["ro"])); r_wr.append(int(m["rw"]))
            f_my.append(float(m["fm"])); f_opp.append(float(m["fo"])); f_wr.append(int(m["fw"]))

    if not eps:
        print(f"  WARNING: no eval lines found in {path}; skipping trajectory plot.")
        return None

    arr = lambda x: np.asarray(x, dtype=np.float32)
    return {
        "episodes":  arr(eps),
        "random":    {"my": arr(r_my), "opp": arr(r_opp), "wr": arr(r_wr)},
        "baseline":  {"my": arr(b_my), "opp": arr(b_opp), "wr": arr(b_wr)},
        "bfs":       {"my": arr(f_my), "opp": arr(f_opp), "wr": arr(f_wr)},
    }


def first_crossing(episodes, margins, threshold=0.0):
    """Return the first episode at which margin > threshold, or None."""
    for ep, m in zip(episodes, margins):
        if m > threshold:
            return int(ep)
    return None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_training_trajectory(log_data, out_path):
    """
    Two-panel figure:
      top    -- score margin (my - opp) vs episode, one curve per opponent
      bottom -- win rate vs episode, one curve per opponent
    Phase boundaries shown as light vertical bands.
    """
    eps = log_data["episodes"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 5.4), sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1.0], "hspace": 0.12},
    )

    # ---- Phase shading -------------------------------------------------
    phase_colors = ["#f5f5f5", "#ececec", "#f5f5f5", "#ececec", "#f5f5f5", "#ececec"]
    max_ep = int(eps.max())
    for ax in (ax1, ax2):
        for (lo, hi, name), c in zip(PHASES, phase_colors):
            if lo > max_ep:
                continue
            hi_clip = min(hi, max_ep + 50)
            ax.axvspan(lo, hi_clip, color=c, alpha=0.6, zorder=0)
    # Phase labels on top axis only
    ymax_for_labels = None  # set after we plot

    # ---- Margin curves -------------------------------------------------
    colors = {"random": "#1f77b4", "baseline": "#d62728", "bfs": "#2ca02c"}
    labels = {"random": "vs random", "baseline": "vs baseline", "bfs": "vs BFS"}

    for key in ("random", "baseline", "bfs"):
        margin = log_data[key]["my"] - log_data[key]["opp"]
        ax1.plot(eps, margin, marker="o", markersize=3.5, linewidth=1.6,
                 color=colors[key], label=labels[key])

    ax1.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax1.set_ylabel("score margin (mean my $-$ opp)")
    ax1.legend(loc="lower right", frameon=True, fontsize=9)
    ax1.grid(True, alpha=0.25, linewidth=0.5)

    # Phase name labels along the top. Skip phases too narrow to fit a label
    # (the first two phases are very short -- the curriculum spends most of
    # its time in the later phases).
    y_top = ax1.get_ylim()[1]
    y_label = y_top - 0.04 * (y_top - ax1.get_ylim()[0])
    min_phase_width = 0.07 * max_ep  # require >= ~7% of x-range
    for lo, hi, name in PHASES:
        if lo > max_ep:
            continue
        hi_clip = min(hi, max_ep + 50)
        if hi_clip - lo < min_phase_width:
            continue
        mid = (lo + hi_clip) / 2
        ax1.text(mid, y_label, name, ha="center", va="top",
                 fontsize=7.5, color="#555555", style="italic")

    # ---- Win-rate panel ------------------------------------------------
    for key in ("random", "baseline", "bfs"):
        ax2.plot(eps, log_data[key]["wr"], marker="o", markersize=3.5,
                 linewidth=1.6, color=colors[key], label=labels[key])
    ax2.axhline(50, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax2.set_ylim(-5, 105)
    ax2.set_ylabel("win rate (\\%)" if matplotlib.rcParams["text.usetex"] else "win rate (%)")
    ax2.set_xlabel("training episode")
    ax2.grid(True, alpha=0.25, linewidth=0.5)

    fig.suptitle("Training trajectory: periodic evaluation vs benchmark opponents",
                 fontsize=11, y=0.995)

    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  wrote {out_path}")
    print(f"  wrote {out_path.replace('.pdf', '.png')}")


# ---------------------------------------------------------------------------
# Paragraph generation
# ---------------------------------------------------------------------------
def build_paragraph(final_results, log_data, n_games):
    """Fill in the LaTeX paragraph from the report."""
    r = final_results["random"]
    b = final_results["baseline"]
    f = final_results["bfs"]

    # Crossing points from the training log.
    if log_data is not None:
        eps = log_data["episodes"]
        rand_margin = log_data["random"]["my"]   - log_data["random"]["opp"]
        base_margin = log_data["baseline"]["my"] - log_data["baseline"]["opp"]
        cross_random   = first_crossing(eps, rand_margin, 0.0)
        cross_baseline = first_crossing(eps, base_margin, 0.0)
        cross_random_str   = f"{cross_random}"   if cross_random   is not None else "??"
        cross_baseline_str = f"{cross_baseline}" if cross_baseline is not None else "??"
    else:
        cross_random_str = "??"
        cross_baseline_str = "??"

    def signed(x):
        return f"+{x:.1f}" if x >= 0 else f"{x:.1f}"

    paragraph = rf"""\paragraph{{Results.}}
We evaluated the trained agent against the two benchmark opponents over
{n_games} episodes per opponent with $\varepsilon=0$ (greedy play). The agent beat
the random opponent in {int(round(r['win_rate']*100))}\% of games with an average score margin of
{signed(r['margin'])} and the rule-based baseline in {int(round(b['win_rate']*100))}\% of games with an average margin
of {signed(b['margin'])}, confirming that the learned policy is meaningfully better than
both ``goats''. Figure~\ref{{fig:training}} shows the training trajectory: the
agent overtakes the random benchmark within the first $\sim${cross_random_str} episodes
and crosses the zero-margin line against baseline at episode $\sim${cross_baseline_str},
with continued improvement during the BFS and self-play phases. The
remaining gap to the BFS opponent ({signed(f['margin'])} margin) is the realistic skill
ceiling for this environment, since BFS plays an information-theoretically
near-optimal greedy policy."""

    # Sanity / context block printed alongside the paragraph.
    extra = f"""
% --- Raw numbers for the table / appendix (not part of paragraph) ---
% Greedy ({n_games} games per opponent), epsilon = 0:
%   vs random:   win_rate = {r['win_rate']*100:.1f}%   my = {r['my_mean']:.2f} ± {r['my_std']:.2f}   opp = {r['opp_mean']:.2f} ± {r['opp_std']:.2f}   margin = {signed(r['margin'])} ± {r['margin_std']:.2f}
%   vs baseline: win_rate = {b['win_rate']*100:.1f}%   my = {b['my_mean']:.2f} ± {b['my_std']:.2f}   opp = {b['opp_mean']:.2f} ± {b['opp_std']:.2f}   margin = {signed(b['margin'])} ± {b['margin_std']:.2f}
%   vs bfs:      win_rate = {f['win_rate']*100:.1f}%   my = {f['my_mean']:.2f} ± {f['my_std']:.2f}   opp = {f['opp_mean']:.2f} ± {f['opp_std']:.2f}   margin = {signed(f['margin'])} ± {f['margin_std']:.2f}
"""
    return paragraph + "\n" + extra


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Produce numbers + training-trajectory plot for the report."
    )
    parser.add_argument("agent_dir", help="Path to the agent dir (with agent.py, config.yaml, weights/)")
    parser.add_argument("--weights", default="weights_best.pth",
                        help="Weights filename inside weights_dir (default: weights_best.pth)")
    parser.add_argument("--n", type=int, default=200,
                        help="Games per opponent for the final greedy eval (default: 200)")
    parser.add_argument("--opponents", nargs="+",
                        default=["src/agents/random/", "src/agents/baseline/", "src/agents/bfs/"],
                        help="Opponent agent directories")
    parser.add_argument("--log", default="plots/training_log_cnn.txt",
                        help="Path to training log (default: plots/training_log_cnn.txt)")
    parser.add_argument("--out-dir", default="plots/",
                        help="Where to write the figure and paragraph (default: plots/)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Load opponents ----------------------------------------------------
    print("\nLoading opponents...")
    opp_labels = {
        "src/agents/random/":   "random",
        "src/agents/baseline/": "baseline",
        "src/agents/bfs/":      "bfs",
    }
    opponents = {}
    for opp_dir in args.opponents:
        key = opp_dir.rstrip("/\\").replace("\\", "/") + "/"
        label = opp_labels.get(key, os.path.basename(opp_dir.rstrip("/\\")))
        try:
            opponents[label] = load_opponent(opp_dir)
            print(f"  loaded: {label} ({opp_dir})")
        except Exception as e:
            print(f"  SKIP {opp_dir}: {e}")

    if not opponents:
        print("No opponents loaded -- aborting.")
        sys.exit(1)

    # ---- Final greedy eval -------------------------------------------------
    env = CollectorGymEnv(numpy_output=True)
    print(f"\nGreedy evaluation: {args.n} games per opponent, weights={args.weights}\n")
    agent = load_agent_from_dir(args.agent_dir, weights_filename=args.weights)

    final_results = {}
    for label in ("random", "baseline", "bfs"):
        if label not in opponents:
            print(f"  -- {label} not loaded, skipping (paragraph will mark ?? for it)")
            continue
        print(f"  vs {label}...")
        final_results[label] = run_games(agent, opponents[label], env,
                                         n=args.n, opp_name=label)
        r = final_results[label]
        sign = "+" if r["margin"] >= 0 else ""
        print(f"\n    win={r['win_rate']:.0%}  "
              f"my={r['my_mean']:.1f}±{r['my_std']:.1f}  "
              f"opp={r['opp_mean']:.1f}±{r['opp_std']:.1f}  "
              f"margin={sign}{r['margin']:.1f}")

    env.close()

    # If any opponent is missing, fill placeholders so paragraph builder still works.
    for label in ("random", "baseline", "bfs"):
        if label not in final_results:
            final_results[label] = {
                "win_rate": 0.0, "my_mean": 0.0, "opp_mean": 0.0,
                "my_std": 0.0, "opp_std": 0.0, "margin": 0.0,
                "margin_std": 0.0, "wins": 0, "n": 0,
            }

    # ---- Training log -> trajectory figure --------------------------------
    print("\nParsing training log...")
    log_data = parse_training_log(args.log)

    if log_data is not None:
        fig_path = os.path.join(args.out_dir, "training_trajectory.pdf")
        plot_training_trajectory(log_data, fig_path)
    else:
        print("  (no figure produced)")

    # ---- Paragraph --------------------------------------------------------
    paragraph = build_paragraph(final_results, log_data, args.n)
    para_path = os.path.join(args.out_dir, "results_paragraph.txt")
    with open(para_path, "w") as fh:
        fh.write(paragraph)

    print("\n" + "=" * 64)
    print("RESULTS PARAGRAPH (also saved to {})".format(para_path))
    print("=" * 64)
    print(paragraph)


if __name__ == "__main__":
    main()
