"""
PPO Agent for the Collector environment.

Inference-time wrapper. Loads a trained CNN actor-critic checkpoint
and selects actions via the policy. By default acts greedily (argmax)
for competition; set deterministic=False in config for sampling.
"""
import os
from types import SimpleNamespace

import numpy as np
import torch

from agents.agent_base import BaseAgent
from environments.collector.state import EnvState

# Local imports - these live next to agent.py in src/agents/agent/
from model import ActorCritic
from preprocessing import encode_observation, NUM_CHANNELS


def _find_project_root(start_path: str) -> str:
    """
    Walk up from start_path until we find a directory containing setup.py.
    Falls back to start_path if not found.
    """
    cur = os.path.abspath(start_path)
    for _ in range(10):  # bounded walk
        if os.path.isfile(os.path.join(cur, "setup.py")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(start_path)


class Agent(BaseAgent):
    """Inference-time PPO agent. See train.py for training."""

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.config = config
        self.action_space = int(getattr(config, "action_space", 4))
        self.deterministic = bool(getattr(config, "deterministic", True))
        self.hidden_dim = int(getattr(config, "hidden_dim", 256))

        # CPU only; small enough that single-thread is fastest
        self.device = torch.device("cpu")
        torch.set_num_threads(1)

        self.network = ActorCritic(
            in_channels=NUM_CHANNELS,
            num_actions=self.action_space,
            hidden_dim=self.hidden_dim,
        ).to(self.device)
        self.network.eval()

        # Resolve weights path. Default: <project_root>/weights/ppo_latest.pt
        # If config gives an absolute path, use it. If relative, resolve
        # relative to the project root (where setup.py lives).
        here = os.path.dirname(os.path.abspath(__file__))
        project_root = _find_project_root(here)
        raw_path = getattr(config, "weights_path", "weights/ppo_latest.pt")
        if os.path.isabs(raw_path):
            self.weights_path = raw_path
        else:
            self.weights_path = os.path.join(project_root, raw_path)

        self._loaded = False

    def load(self) -> None:
        """Load trained weights. Silent if file missing (acts as random init)."""
        if not os.path.exists(self.weights_path):
            print(f"[ppo agent] WARNING: weights not found at {self.weights_path}, "
                  f"using random init. Train with train.py first.")
            return
        try:
            ckpt = torch.load(self.weights_path, map_location=self.device, weights_only=True)
        except TypeError:
            # Older torch versions don't have weights_only kwarg
            ckpt = torch.load(self.weights_path, map_location=self.device)

        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
        self.network.load_state_dict(state_dict)
        self.network.eval()
        self._loaded = True

    def act(self, observation: EnvState) -> int:
        feat = encode_observation(observation)  # (C, H, W) float32
        x = torch.from_numpy(feat).unsqueeze(0).to(self.device)
        action, _, _ = self.network.act(x, deterministic=self.deterministic)
        return int(action.item())