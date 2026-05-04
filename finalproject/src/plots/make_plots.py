"""
Parse training log and produce report-quality plots.

Generates four plots:
  1. Training progress: avg score per 50-ep window vs episode (with epsilon overlay)
  2. Eval performance: agent vs baseline score over training (with margin)
  3. Eval performance: agent vs random score over training
  4. Combined dashboard: all key metrics on one figure for the report
"""
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------- parsing ----------
LOG_PATH = "training_log.txt"
OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

ep_pattern = re.compile(
    r"Ep\s+(\d+)\s+\|\s+opp=(\S+)\s+\|\s+avg_raw_reward=\s*(-?[\d.]+)\s+\|\s+"
    r"avg_score=([\d.]+)\s+\|\s+eps=([\d.]+)\s+\|\s+buf=(\d+)\s+\|\s+dshape=([\d.]+)"
)
eval_pattern = re.compile(
    r"\[eval\]\s+vs\s+baseline:\s+([\d.]+)\s+-\s+([\d.]+)\s+win_rate=([\d.]+)%\s+\|\s+"
    r"vs\s+random:\s+([\d.]+)\s+-\s+([\d.]+)\s+win_rate=([\d.]+)%"
)

train_episodes = []
train_scores = []
train_rewards = []
train_eps = []
train_dshape = []
train_opponent = []

eval_episodes = []
eval_b_my, eval_b_opp, eval_b_wr = [], [], []
eval_r_my, eval_r_opp, eval_r_wr = [], [], []

last_train_ep = 0
with open(LOG_PATH) as f:
    for line in f:
        m = ep_pattern.search(line)
        if m:
            ep = int(m.group(1))
            train_episodes.append(ep)
            train_opponent.append(m.group(2))
            train_rewards.append(float(m.group(3)))
            train_scores.append(float(m.group(4)))
            train_eps.append(float(m.group(5)))
            train_dshape.append(float(m.group(7)))
            last_train_ep = ep
            continue
        m = eval_pattern.search(line)
        if m:
            # Eval is reported just before the next training Ep print, so use last_train_ep
            eval_episodes.append(last_train_ep)
            eval_b_my.append(float(m.group(1)))
            eval_b_opp.append(float(m.group(2)))
            eval_b_wr.append(float(m.group(3)))
            eval_r_my.append(float(m.group(4)))
            eval_r_opp.append(float(m.group(5)))
            eval_r_wr.append(float(m.group(6)))

train_episodes = np.array(train_episodes)
train_scores = np.array(train_scores)
train_rewards = np.array(train_rewards)
train_eps = np.array(train_eps)
train_dshape = np.array(train_dshape)

eval_episodes = np.array(eval_episodes)
eval_b_my = np.array(eval_b_my)
eval_b_opp = np.array(eval_b_opp)
eval_b_wr = np.array(eval_b_wr)
eval_r_my = np.array(eval_r_my)
eval_r_opp = np.array(eval_r_opp)
eval_r_wr = np.array(eval_r_wr)

print(f"Parsed {len(train_episodes)} training rows and {len(eval_episodes)} eval rows.")

# ---------- styling ----------
plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.dpi": 110,
})

AGENT_COLOR = "#2E86DE"
BASELINE_COLOR = "#E74C3C"
RANDOM_COLOR = "#7F8C8D"
EPS_COLOR = "#F39C12"
WIN_COLOR = "#27AE60"


# ---------- plot 1: training progress ----------
fig, ax1 = plt.subplots(figsize=(9, 4.5))
ax1.plot(train_episodes, train_scores, color=AGENT_COLOR, linewidth=1.6,
         label="Avg score (per 50-episode window)")
ax1.fill_between(train_episodes, train_scores, alpha=0.15, color=AGENT_COLOR)
ax1.set_xlabel("Training episode")
ax1.set_ylabel("Average items collected", color=AGENT_COLOR)
ax1.tick_params(axis='y', labelcolor=AGENT_COLOR)
ax1.set_ylim(0, max(train_scores) * 1.15)

ax2 = ax1.twinx()
ax2.plot(train_episodes, train_eps, color=EPS_COLOR, linewidth=1.4,
         linestyle="--", label="Epsilon")
ax2.set_ylabel("Epsilon (exploration rate)", color=EPS_COLOR)
ax2.tick_params(axis='y', labelcolor=EPS_COLOR)
ax2.set_ylim(0, 1.05)
ax2.spines["top"].set_visible(False)
ax2.grid(False)

ax1.set_title("Training progress: agent score and exploration over time")

# Curriculum phase markers
ax1.axvline(500, color="black", alpha=0.15, linewidth=1)
ax1.axvline(1500, color="black", alpha=0.15, linewidth=1)
ax1.axvline(3000, color="black", alpha=0.15, linewidth=1)
ax1.text(250, ax1.get_ylim()[1] * 0.95, "Phase 1\n50/50 rand/base",
         ha="center", fontsize=8, alpha=0.6)
ax1.text(1000, ax1.get_ylim()[1] * 0.95, "Phase 2\nmostly baseline",
         ha="center", fontsize=8, alpha=0.6)
ax1.text(2250, ax1.get_ylim()[1] * 0.95, "Phase 3\n+ self-play",
         ha="center", fontsize=8, alpha=0.6)
ax1.text(3500, ax1.get_ylim()[1] * 0.95, "Phase 4\nheavy self-play",
         ha="center", fontsize=8, alpha=0.6)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "1_training_progress.png"), dpi=160, bbox_inches="tight")
print("saved 1_training_progress.png")
plt.close(fig)


# ---------- plot 2: eval vs baseline ----------
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5))

ax_a.plot(eval_episodes, eval_b_my, "o-", color=AGENT_COLOR, linewidth=2,
          markersize=6, label="Our agent")
ax_a.plot(eval_episodes, eval_b_opp, "s-", color=BASELINE_COLOR, linewidth=2,
          markersize=6, label="Baseline opponent")
ax_a.fill_between(eval_episodes, eval_b_my, eval_b_opp,
                  where=eval_b_my >= eval_b_opp, alpha=0.15, color=AGENT_COLOR,
                  label="Our lead")
ax_a.set_xlabel("Training episode")
ax_a.set_ylabel("Items collected (avg over 10 eval games)")
ax_a.set_title("Evaluation: our agent vs baseline")
ax_a.legend(loc="lower right", framealpha=0.95)
ax_a.set_ylim(0, max(eval_b_my.max(), eval_b_opp.max()) * 1.15)

ax_b.plot(eval_episodes, eval_b_wr, "o-", color=WIN_COLOR, linewidth=2, markersize=6)
ax_b.fill_between(eval_episodes, 0, eval_b_wr, alpha=0.2, color=WIN_COLOR)
ax_b.axhline(50, color="black", linestyle=":", alpha=0.5, linewidth=1, label="50% (toss-up)")
ax_b.set_xlabel("Training episode")
ax_b.set_ylabel("Win rate vs baseline (%)")
ax_b.set_title("Win rate vs baseline (10-game average)")
ax_b.set_ylim(0, 105)
ax_b.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax_b.legend(loc="lower right", framealpha=0.95)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "2_eval_vs_baseline.png"), dpi=160, bbox_inches="tight")
print("saved 2_eval_vs_baseline.png")
plt.close(fig)


# ---------- plot 3: eval vs random (sanity check) ----------
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(eval_episodes, eval_r_my, "o-", color=AGENT_COLOR, linewidth=2,
        markersize=6, label="Our agent")
ax.plot(eval_episodes, eval_r_opp, "s-", color=RANDOM_COLOR, linewidth=2,
        markersize=6, label="Random opponent")
ax.fill_between(eval_episodes, eval_r_my, eval_r_opp,
                where=eval_r_my >= eval_r_opp, alpha=0.15, color=AGENT_COLOR)
ax.set_xlabel("Training episode")
ax.set_ylabel("Items collected (avg over 10 eval games)")
ax.set_title("Evaluation: our agent vs random (sanity check)")
ax.legend(loc="center right", framealpha=0.95)
ax.set_ylim(0, max(eval_r_my.max(), eval_r_opp.max()) * 1.15)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "3_eval_vs_random.png"), dpi=160, bbox_inches="tight")
print("saved 3_eval_vs_random.png")
plt.close(fig)


# ---------- plot 4: combined report dashboard ----------
fig = plt.figure(figsize=(13, 7.5))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.30)

# top-left: training score + epsilon
ax_tl = fig.add_subplot(gs[0, 0])
ax_tl.plot(train_episodes, train_scores, color=AGENT_COLOR, linewidth=1.6)
ax_tl.fill_between(train_episodes, train_scores, alpha=0.15, color=AGENT_COLOR)
ax_tl.set_xlabel("Episode")
ax_tl.set_ylabel("Avg items / 50-ep window", color=AGENT_COLOR)
ax_tl.tick_params(axis='y', labelcolor=AGENT_COLOR)
ax_tl.set_title("(a) Training score")
ax_tl_2 = ax_tl.twinx()
ax_tl_2.plot(train_episodes, train_eps, color=EPS_COLOR, linewidth=1.2, linestyle="--")
ax_tl_2.set_ylabel("Epsilon", color=EPS_COLOR)
ax_tl_2.tick_params(axis='y', labelcolor=EPS_COLOR)
ax_tl_2.set_ylim(0, 1.05)
ax_tl_2.grid(False)

# top-right: eval vs baseline scores
ax_tr = fig.add_subplot(gs[0, 1])
ax_tr.plot(eval_episodes, eval_b_my, "o-", color=AGENT_COLOR, linewidth=2,
           markersize=5, label="Our agent")
ax_tr.plot(eval_episodes, eval_b_opp, "s-", color=BASELINE_COLOR, linewidth=2,
           markersize=5, label="Baseline")
ax_tr.fill_between(eval_episodes, eval_b_my, eval_b_opp,
                   where=eval_b_my >= eval_b_opp, alpha=0.15, color=AGENT_COLOR)
ax_tr.set_xlabel("Episode")
ax_tr.set_ylabel("Items collected")
ax_tr.set_title("(b) Eval vs baseline (10-game avg)")
ax_tr.legend(loc="lower right", framealpha=0.95, fontsize=9)
ax_tr.set_ylim(0, max(eval_b_my.max(), eval_b_opp.max()) * 1.15)

# bottom-left: win rates (both opponents)
ax_bl = fig.add_subplot(gs[1, 0])
ax_bl.plot(eval_episodes, eval_b_wr, "o-", color=BASELINE_COLOR, linewidth=2,
           markersize=5, label="vs baseline")
ax_bl.plot(eval_episodes, eval_r_wr, "s-", color=RANDOM_COLOR, linewidth=2,
           markersize=5, label="vs random")
ax_bl.axhline(50, color="black", linestyle=":", alpha=0.5, linewidth=1)
ax_bl.set_xlabel("Episode")
ax_bl.set_ylabel("Win rate (%)")
ax_bl.set_title("(c) Win rate over training")
ax_bl.legend(loc="lower right", framealpha=0.95, fontsize=9)
ax_bl.set_ylim(0, 105)
ax_bl.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))

# bottom-right: final-performance bars (last eval and best eval)
ax_br = fig.add_subplot(gs[1, 1])
last_idx = -1
best_idx = int(np.argmax(eval_b_my - eval_b_opp))
labels = ["Best eval", "Final eval"]
our_vals = [eval_b_my[best_idx], eval_b_my[last_idx]]
their_vals = [eval_b_opp[best_idx], eval_b_opp[last_idx]]
random_their_vals = [eval_r_opp[best_idx], eval_r_opp[last_idx]]
random_my_vals = [eval_r_my[best_idx], eval_r_my[last_idx]]

x = np.arange(len(labels))
width = 0.2

ax_br.bar(x - 1.5*width, our_vals, width, color=AGENT_COLOR, label="Us (vs baseline)")
ax_br.bar(x - 0.5*width, their_vals, width, color=BASELINE_COLOR, label="Baseline")
ax_br.bar(x + 0.5*width, random_my_vals, width, color=AGENT_COLOR, alpha=0.5,
          label="Us (vs random)")
ax_br.bar(x + 1.5*width, random_their_vals, width, color=RANDOM_COLOR, label="Random")
for i, v in enumerate(our_vals):
    ax_br.text(x[i] - 1.5*width, v + 2, f"{v:.0f}", ha="center", fontsize=8, color=AGENT_COLOR)
for i, v in enumerate(their_vals):
    ax_br.text(x[i] - 0.5*width, v + 2, f"{v:.0f}", ha="center", fontsize=8, color=BASELINE_COLOR)
for i, v in enumerate(random_my_vals):
    ax_br.text(x[i] + 0.5*width, v + 2, f"{v:.0f}", ha="center", fontsize=8, color=AGENT_COLOR, alpha=0.7)
for i, v in enumerate(random_their_vals):
    ax_br.text(x[i] + 1.5*width, v + 2, f"{v:.0f}", ha="center", fontsize=8, color=RANDOM_COLOR)
ax_br.set_xticks(x)
ax_br.set_xticklabels(labels)
ax_br.set_ylabel("Items collected")
ax_br.set_title("(d) Best vs final evaluation")
ax_br.legend(loc="upper right", framealpha=0.95, fontsize=8)
ax_br.set_ylim(0, max(max(our_vals), max(their_vals)) * 1.25)

fig.suptitle("DQN agent training summary — Collector environment",
             fontsize=14, fontweight="bold", y=0.995)
fig.savefig(os.path.join(OUT_DIR, "4_dashboard.png"), dpi=160, bbox_inches="tight")
print("saved 4_dashboard.png")
plt.close(fig)

# ---------- summary printout ----------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Training episodes:        {train_episodes[-1]}")
print(f"Final training score:     {train_scores[-1]:.2f} items / episode")
print(f"Best eval vs baseline:    {eval_b_my[best_idx]:.1f} - {eval_b_opp[best_idx]:.1f} "
      f"(margin: +{eval_b_my[best_idx]-eval_b_opp[best_idx]:.1f}) at ep {eval_episodes[best_idx]}")
print(f"Final eval vs baseline:   {eval_b_my[-1]:.1f} - {eval_b_opp[-1]:.1f} "
      f"(margin: +{eval_b_my[-1]-eval_b_opp[-1]:.1f})")
print(f"Avg win rate vs baseline (last 10 evals):  {eval_b_wr[-10:].mean():.1f}%")
print(f"Avg win rate vs random   (last 10 evals):  {eval_r_wr[-10:].mean():.1f}%")