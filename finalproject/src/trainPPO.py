"""
PPO trainer for the Collector environment.

Run from project root:
    python src/agents/agent/train.py
or with a custom config:
    python src/agents/agent/train.py --config src/agents/agent/config.yaml
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import os
import random
import sys
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import yaml

# Environment
from environments.collector.wrappers import CollectorGymEnv
from environments.collector.params import EnvParams

# Local imports - same-dir style (matches your agent.py)
from agents.agent_ppo.model import ActorCritic
from agents.agent_ppo.preprocessing import (
    encode_observation, shaping_potential, NUM_CHANNELS,
)


# ============================================================================
# Project root resolution + on-disk agent loader (mirrors compete.py)
# ============================================================================
def find_project_root(start_path: str) -> str:
    cur = os.path.abspath(start_path)
    for _ in range(10):
        if os.path.isfile(os.path.join(cur, "setup.py")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(start_path)


class OpponentFactory:
    """
    Creates fresh opponent instances on demand. Mirrors tournament behavior.
    """

    def __init__(self):
        self._cache = {}

    def _ensure_loaded(self, agent_dir: str):
        if agent_dir in self._cache:
            return
        config_path = os.path.join(agent_dir, "config.yaml")
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}
        agent_file = os.path.join(agent_dir, "agent.py")
        mod_name = f"_oppfactory_{os.path.basename(os.path.normpath(agent_dir))}"
        spec = importlib.util.spec_from_file_location(mod_name, agent_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._cache[agent_dir] = (config_dict, module.Agent)

    def make(self, agent_dir: str, seed_override: int | None = None):
        self._ensure_loaded(agent_dir)
        config_dict, AgentClass = self._cache[agent_dir]
        cfg_copy = dict(config_dict)
        if seed_override is not None and "seed" in cfg_copy:
            cfg_copy["seed"] = int(seed_override)

        class _Cfg:
            pass

        cfg_obj = _Cfg()
        for k, v in cfg_copy.items():
            setattr(cfg_obj, k, v)

        agent = AgentClass(cfg_obj)
        agent.load()
        return agent


class SnapshotPolicy:
    def __init__(self, network: ActorCritic, deterministic: bool = False):
        self.network = network
        self.deterministic = deterministic
        self.network.eval()

    def act(self, obs: dict) -> int:
        feat = encode_observation(obs)
        x = torch.from_numpy(feat).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = self.network.act(x, deterministic=self.deterministic)
        return int(action.item())


# ============================================================================
# Per-env wrapper
# ============================================================================
class TrainingEnv:
    def __init__(self, seed: int, min_steps: int, max_steps: int):
        self.env = CollectorGymEnv(numpy_output=True)
        self.params = EnvParams()
        self.rng = np.random.default_rng(seed)
        self.min_steps = min_steps
        self.max_steps = max_steps

        self.opponent = None
        self.shaping_coef = 0.0
        self.reward_scale = 1.0

        self._obs = None
        self._ep_step = 0
        self._ep_horizon = max_steps
        self._ep_return = 0.0
        self._ep_shaped_return = 0.0
        self._ep_items = 0
        self._ep_opp_items = 0
        self._ep_crashes = 0
        self._prev_potential = 0.0
        self._prev_team_points = None

    def set_opponent(self, opponent):
        self.opponent = opponent

    def set_shaping_coef(self, c: float):
        self.shaping_coef = float(c)

    def set_reward_scale(self, s: float):
        self.reward_scale = float(s)

    def _new_horizon(self) -> int:
        return int(self.rng.integers(self.min_steps, self.max_steps + 1))

    def reset(self):
        seed = int(self.rng.integers(0, 2**31 - 1))
        obs, info = self.env.reset(seed=seed, options=dict(params=self.params))
        self._obs = obs
        self._ep_step = 0
        self._ep_horizon = self._new_horizon()
        self._ep_return = 0.0
        self._ep_shaped_return = 0.0
        self._ep_items = 0
        self._ep_opp_items = 0
        self._ep_crashes = 0
        self._prev_potential = shaping_potential(obs["player_0"])
        # Track team points to count items reliably (reward when collecting
        # an item is 0, not 1, because of the per-step -1 cost)
        self._prev_team_points = np.asarray(
            obs["player_0"]["team_points"], dtype=np.int32
        ).copy()
        return encode_observation(obs["player_0"])

    def step(self, action: int):
        opp_action = self.opponent.act(self._obs["player_1"])
        actions = {"player_0": int(action), "player_1": int(opp_action)}
        next_obs, reward, terminated, truncated, info = self.env.step(actions)

        r0 = float(reward[0])
        r1 = float(reward[1])

        # Item counting via team_points delta (player_0 perspective:
        # team_points[0] = me, [1] = opp)
        new_tp = np.asarray(next_obs["player_0"]["team_points"], dtype=np.int32)
        delta_me = int(new_tp[0] - self._prev_team_points[0])
        delta_opp = int(new_tp[1] - self._prev_team_points[1])
        if delta_me > 0:
            self._ep_items += delta_me
        if delta_opp > 0:
            self._ep_opp_items += delta_opp
        self._prev_team_points = new_tp.copy()

        # Wall crash: r0 == -2 means we tried to move into a wall/obstacle
        if r0 <= -1.5:
            self._ep_crashes += 1

        # Potential-based shaping
        new_potential = shaping_potential(next_obs["player_0"])
        shaped_bonus = self.shaping_coef * (new_potential - self._prev_potential)
        self._prev_potential = new_potential

        shaped_reward = r0 + shaped_bonus
        self._ep_return += r0
        self._ep_shaped_return += shaped_reward
        self._ep_step += 1

        # Scale the reward going into the PPO buffer (keep ep_return raw for logging).
        # Smaller reward magnitudes -> smaller value targets -> easier value
        # function fitting -> policy gradient gets a real signal.
        scaled_reward = shaped_reward * self.reward_scale

        env_done = bool(terminated) or bool(truncated)
        horizon_done = self._ep_step >= self._ep_horizon
        done = env_done or horizon_done

        ep_info = None
        if done:
            ep_info = dict(
                ep_return=self._ep_return,
                ep_shaped=self._ep_shaped_return,
                ep_len=self._ep_step,
                ep_items=self._ep_items,
                ep_opp_items=self._ep_opp_items,
                ep_crashes=self._ep_crashes,
                ep_horizon=self._ep_horizon,
            )

        if done:
            next_encoded = self.reset()
        else:
            self._obs = next_obs
            next_encoded = encode_observation(next_obs["player_0"])

        return next_encoded, scaled_reward, done, ep_info


# ============================================================================
# Rollout buffer
# ============================================================================
class RolloutBuffer:
    def __init__(self, num_envs: int, rollout_steps: int, obs_shape, device):
        self.num_envs = num_envs
        self.T = rollout_steps
        self.device = device
        self.obs = torch.zeros((self.T, num_envs, *obs_shape), dtype=torch.float32)
        self.actions = torch.zeros((self.T, num_envs), dtype=torch.long)
        self.logprobs = torch.zeros((self.T, num_envs), dtype=torch.float32)
        self.rewards = torch.zeros((self.T, num_envs), dtype=torch.float32)
        self.dones = torch.zeros((self.T, num_envs), dtype=torch.float32)
        self.values = torch.zeros((self.T, num_envs), dtype=torch.float32)

    def compute_gae(self, last_values, last_dones, gamma, gae_lambda):
        advantages = torch.zeros_like(self.rewards)
        last_gae = torch.zeros(self.num_envs, dtype=torch.float32)
        for t in reversed(range(self.T)):
            if t == self.T - 1:
                next_nonterminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_nonterminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_values * next_nonterminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values
        return advantages, returns


# ============================================================================
# Curriculum
# ============================================================================
def make_opponent_for_phase(progress: float, cfg: SimpleNamespace,
                            project_root: str, factory: OpponentFactory,
                            snapshot_pool, rng) -> object:
    p_rand = cfg.phase_random_frac
    p_base = p_rand + cfg.phase_baseline_frac
    p_bfs = p_base + cfg.phase_bfs_frac
    paths = cfg.opponent_paths

    def _abspath(p):
        return p if os.path.isabs(p) else os.path.join(project_root, p)

    seed = int(rng.integers(0, 2**31 - 1))

    if progress < p_rand:
        return factory.make(_abspath(paths["random"]), seed_override=seed)
    if progress < p_base:
        return factory.make(_abspath(paths["baseline"]), seed_override=seed)
    if progress < p_bfs:
        return factory.make(_abspath(paths["bfs"]), seed_override=seed)

    if not snapshot_pool or rng.random() < cfg.selfplay_bfs_mix:
        return factory.make(_abspath(paths["bfs"]), seed_override=seed)
    snap = snapshot_pool[rng.integers(0, len(snapshot_pool))]
    return SnapshotPolicy(snap, deterministic=False)


# ============================================================================
# Evaluation
# ============================================================================
def evaluate_against(network, opponent_factory_callable, num_episodes: int,
                     max_steps: int = 1000, seed_base: int = 100000):
    network.eval()
    env = CollectorGymEnv(numpy_output=True)
    params = EnvParams()
    wins = ties = losses = 0
    items = []
    opp_items = []
    returns = []

    for ep in range(num_episodes):
        opp = opponent_factory_callable()
        obs, info = env.reset(seed=seed_base + ep, options=dict(params=params))
        ep_ret = 0.0
        prev_tp = np.asarray(obs["player_0"]["team_points"], dtype=np.int32).copy()
        ep_items = 0
        ep_opp_items = 0
        for _ in range(max_steps):
            feat = encode_observation(obs["player_0"])
            x = torch.from_numpy(feat).unsqueeze(0)
            with torch.no_grad():
                a, _, _ = network.act(x, deterministic=True)
            opp_a = opp.act(obs["player_1"])
            actions = {"player_0": int(a.item()), "player_1": int(opp_a)}
            obs, reward, terminated, truncated, info = env.step(actions)
            r0 = float(reward[0])
            ep_ret += r0
            new_tp = np.asarray(obs["player_0"]["team_points"], dtype=np.int32)
            ep_items += max(0, int(new_tp[0] - prev_tp[0]))
            ep_opp_items += max(0, int(new_tp[1] - prev_tp[1]))
            prev_tp = new_tp.copy()
            if terminated or truncated:
                break

        if ep_items > ep_opp_items: wins += 1
        elif ep_items == ep_opp_items: ties += 1
        else: losses += 1
        items.append(ep_items)
        opp_items.append(ep_opp_items)
        returns.append(ep_ret)

    return dict(
        win_rate=wins / num_episodes,
        tie_rate=ties / num_episodes,
        loss_rate=losses / num_episodes,
        mean_items=float(np.mean(items)),
        mean_opp_items=float(np.mean(opp_items)),
        mean_return=float(np.mean(returns)),
    )


# ============================================================================
# Main training loop
# ============================================================================
def load_config(path: str) -> SimpleNamespace:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    return SimpleNamespace(**d)


def train(config_path: str):
    cfg = load_config(config_path)

    seed = int(getattr(cfg, "seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    device = torch.device("cpu")
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

    here = os.path.dirname(os.path.abspath(__file__))
    project_root = find_project_root(here)
    print(f"[train] project root: {project_root}", flush=True)

    save_dir = os.path.join(project_root, "weights")
    os.makedirs(save_dir, exist_ok=True)
    latest_path = os.path.join(save_dir, "ppo_latest.pt")
    print(f"[train] weights dir: {save_dir}", flush=True)

    for name, rel in cfg.opponent_paths.items():
        full = rel if os.path.isabs(rel) else os.path.join(project_root, rel)
        if not (os.path.isfile(os.path.join(full, "agent.py")) and
                os.path.isfile(os.path.join(full, "config.yaml"))):
            raise FileNotFoundError(f"Opponent '{name}' missing files in {full}")
    print(f"[train] opponents OK: {list(cfg.opponent_paths.keys())}", flush=True)

    factory = OpponentFactory()

    network = ActorCritic(
        in_channels=NUM_CHANNELS,
        num_actions=4,
        hidden_dim=int(cfg.hidden_dim),
    ).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=float(cfg.learning_rate),
                                 eps=float(cfg.adam_eps))

    num_envs = int(cfg.num_envs)
    rollout_steps = int(cfg.rollout_steps)
    rng = np.random.default_rng(seed)
    envs = [TrainingEnv(seed=seed + 1000 + i,
                        min_steps=int(cfg.min_episode_steps),
                        max_steps=int(cfg.max_episode_steps))
            for i in range(num_envs)]

    snapshot_pool = []
    for env in envs:
        env.set_opponent(make_opponent_for_phase(
            0.0, cfg, project_root, factory, snapshot_pool, rng,
        ))
        env.set_shaping_coef(float(cfg.shaping_coef_initial))
        env.set_reward_scale(float(getattr(cfg, "reward_scale", 0.2)))

    obs_shape = (NUM_CHANNELS, 16, 16)
    cur_obs = torch.zeros((num_envs, *obs_shape), dtype=torch.float32)
    for i, env in enumerate(envs):
        cur_obs[i] = torch.from_numpy(env.reset())
    cur_dones = torch.zeros(num_envs, dtype=torch.float32)

    buf = RolloutBuffer(num_envs, rollout_steps, obs_shape, device)

    recent_returns = deque(maxlen=100)
    recent_items = deque(maxlen=100)
    recent_opp_items = deque(maxlen=100)
    recent_lens = deque(maxlen=100)
    recent_crashes = deque(maxlen=100)

    total_updates = int(cfg.total_updates)
    global_step = 0
    train_start = time.time()

    for update in range(1, total_updates + 1):
        progress = (update - 1) / total_updates

        ent_coef = float(cfg.entropy_coef) + (
            float(cfg.entropy_coef_final) - float(cfg.entropy_coef)
        ) * progress
        shaping_coef = float(cfg.shaping_coef_initial) + (
            float(cfg.shaping_coef_final) - float(cfg.shaping_coef_initial)
        ) * progress
        for env in envs:
            env.set_shaping_coef(shaping_coef)

        for env in envs:
            if rng.random() < 0.25:
                env.set_opponent(make_opponent_for_phase(
                    progress, cfg, project_root, factory, snapshot_pool, rng,
                ))

        # ============ Rollout ============
        network.eval()
        for t in range(rollout_steps):
            buf.obs[t] = cur_obs
            buf.dones[t] = cur_dones

            with torch.no_grad():
                actions, logprobs, values = network.act(cur_obs, deterministic=False)
            buf.actions[t] = actions
            buf.logprobs[t] = logprobs
            buf.values[t] = values

            next_obs_np = np.zeros((num_envs, *obs_shape), dtype=np.float32)
            rewards_np = np.zeros(num_envs, dtype=np.float32)
            dones_np = np.zeros(num_envs, dtype=np.float32)
            for i, env in enumerate(envs):
                obs_i, r_i, d_i, ep_info = env.step(int(actions[i].item()))
                next_obs_np[i] = obs_i
                rewards_np[i] = r_i
                dones_np[i] = float(d_i)
                if ep_info is not None:
                    recent_returns.append(ep_info["ep_return"])
                    recent_items.append(ep_info["ep_items"])
                    recent_opp_items.append(ep_info["ep_opp_items"])
                    recent_lens.append(ep_info["ep_len"])
                    recent_crashes.append(ep_info["ep_crashes"])
                    if progress >= cfg.phase_random_frac + cfg.phase_baseline_frac:
                        if rng.random() < 0.5:
                            env.set_opponent(make_opponent_for_phase(
                                progress, cfg, project_root, factory,
                                snapshot_pool, rng,
                            ))
            buf.rewards[t] = torch.from_numpy(rewards_np)
            cur_obs = torch.from_numpy(next_obs_np)
            cur_dones = torch.from_numpy(dones_np)
            global_step += num_envs

        with torch.no_grad():
            _, _, last_values = network.act(cur_obs, deterministic=False)
        advantages, returns = buf.compute_gae(
            last_values, cur_dones,
            gamma=float(cfg.gamma), gae_lambda=float(cfg.gae_lambda),
        )

        # ============ PPO update ============
        network.train()
        b_obs = buf.obs.reshape(-1, *obs_shape)
        b_actions = buf.actions.reshape(-1)
        b_logprobs = buf.logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = buf.values.reshape(-1)

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        batch_size = b_obs.shape[0]
        minibatch_size = int(cfg.minibatch_size)
        clip_eps = float(cfg.clip_eps)
        max_grad_norm = float(cfg.max_grad_norm)
        target_kl = float(cfg.target_kl)
        vf_coef = float(cfg.value_loss_coef)

        indices = np.arange(batch_size)
        approx_kls = []
        early_stop = False
        last_pi_loss = last_v_loss = last_ent = 0.0
        for epoch in range(int(cfg.ppo_epochs)):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]

                new_logprobs, entropy, new_values = network.evaluate_actions(
                    b_obs[mb_idx], b_actions[mb_idx]
                )
                logratio = new_logprobs - b_logprobs[mb_idx]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    approx_kls.append(approx_kl)

                mb_adv = b_advantages[mb_idx]
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
                pg_loss = torch.max(pg1, pg2).mean()

                v_clipped = b_values[mb_idx] + torch.clamp(
                    new_values - b_values[mb_idx], -clip_eps, clip_eps
                )
                v_loss1 = (new_values - b_returns[mb_idx]) ** 2
                v_loss2 = (v_clipped - b_returns[mb_idx]) ** 2
                v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

                ent_loss = entropy.mean()
                loss = pg_loss + vf_coef * v_loss - ent_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm)
                optimizer.step()

                last_pi_loss = pg_loss.item()
                last_v_loss = v_loss.item()
                last_ent = ent_loss.item()

            mb_per_epoch = max(1, batch_size // minibatch_size)
            mean_kl = float(np.mean(approx_kls[-mb_per_epoch:]))
            if mean_kl > target_kl:
                early_stop = True
                break

        # ============ Logging ============
        if update % int(cfg.log_every) == 0:
            mean_ret = float(np.mean(recent_returns)) if recent_returns else 0.0
            mean_items = float(np.mean(recent_items)) if recent_items else 0.0
            mean_opp = float(np.mean(recent_opp_items)) if recent_opp_items else 0.0
            mean_len = float(np.mean(recent_lens)) if recent_lens else 0.0
            mean_crash = float(np.mean(recent_crashes)) if recent_crashes else 0.0
            elapsed = time.time() - train_start
            fps = global_step / max(1.0, elapsed)
            kl_str = f"{np.mean(approx_kls):.4f}" if approx_kls else "n/a"
            stop_marker = " [KL-stop]" if early_stop else ""
            print(
                f"[upd {update:4d}/{total_updates}] "
                f"steps={global_step:>9d} fps={fps:5.0f} "
                f"items={mean_items:4.1f}-{mean_opp:4.1f} crash={mean_crash:4.0f} "
                f"ret={mean_ret:+6.0f} ep_len={mean_len:5.0f} | "
                f"pi={last_pi_loss:+.3f} v={last_v_loss:.2f} ent={last_ent:.2f} "
                f"kl={kl_str} ent_c={ent_coef:.3f} shp={shaping_coef:.3f}"
                f"{stop_marker}",
                flush=True,
            )

        if update % int(cfg.snapshot_every) == 0:
            snapshot = ActorCritic(
                in_channels=NUM_CHANNELS, num_actions=4,
                hidden_dim=int(cfg.hidden_dim),
            )
            snapshot.load_state_dict(copy.deepcopy(network.state_dict()))
            snapshot.eval()
            snapshot_pool.append(snapshot)
            if len(snapshot_pool) > int(cfg.selfplay_pool_size):
                snapshot_pool.pop(0)

        if update % int(cfg.eval_every) == 0:
            print(f"  -- evaluating at update {update} --", flush=True)
            eval_t0 = time.time()
            eval_episodes = int(cfg.eval_episodes)

            paths = cfg.opponent_paths

            def _abs(p):
                return p if os.path.isabs(p) else os.path.join(project_root, p)

            r_random = evaluate_against(
                network,
                lambda: factory.make(_abs(paths["random"]),
                                     seed_override=int(rng.integers(0, 2**31 - 1))),
                eval_episodes, seed_base=200000 + update * 100,
            )
            r_baseline = evaluate_against(
                network,
                lambda: factory.make(_abs(paths["baseline"]),
                                     seed_override=int(rng.integers(0, 2**31 - 1))),
                eval_episodes, seed_base=300000 + update * 100,
            )
            r_bfs = evaluate_against(
                network,
                lambda: factory.make(_abs(paths["bfs"]),
                                     seed_override=int(rng.integers(0, 2**31 - 1))),
                eval_episodes, seed_base=400000 + update * 100,
            )
            print(
                f"  vs random:   win={r_random['win_rate']*100:5.1f}% "
                f"items={r_random['mean_items']:5.1f}-{r_random['mean_opp_items']:5.1f} "
                f"ret={r_random['mean_return']:+6.0f}",
                flush=True,
            )
            print(
                f"  vs baseline: win={r_baseline['win_rate']*100:5.1f}% "
                f"items={r_baseline['mean_items']:5.1f}-{r_baseline['mean_opp_items']:5.1f} "
                f"ret={r_baseline['mean_return']:+6.0f}",
                flush=True,
            )
            print(
                f"  vs bfs:      win={r_bfs['win_rate']*100:5.1f}% "
                f"items={r_bfs['mean_items']:5.1f}-{r_bfs['mean_opp_items']:5.1f} "
                f"ret={r_bfs['mean_return']:+6.0f} "
                f"(eval_time={time.time()-eval_t0:.1f}s)",
                flush=True,
            )

        if update % int(cfg.checkpoint_every) == 0 or update == total_updates:
            torch.save({"model": network.state_dict(), "update": update,
                        "config": vars(cfg)}, latest_path)
            ckpt_path = os.path.join(save_dir, f"ppo_upd{update}.pt")
            torch.save({"model": network.state_dict(), "update": update,
                        "config": vars(cfg)}, ckpt_path)
            print(f"  checkpoint saved: {latest_path} (and {ckpt_path})", flush=True)

    torch.save({"model": network.state_dict(), "update": total_updates,
                "config": vars(cfg)}, latest_path)
    print(f"\nTraining complete. Final weights -> {latest_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--config", type=str,
        default=os.path.join("src/agents/agent_ppo", "config.yaml"),
        help="Path to config.yaml",
    )
    args = parser.parse_args()
    train(args.config)