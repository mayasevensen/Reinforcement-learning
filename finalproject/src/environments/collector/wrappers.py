import json
import os
from typing import Any, SupportsFloat
import flax
import flax.serialization
import gymnasium as gym
import gymnax
import gymnax.environments.spaces
import jax
import numpy as np
import dataclasses
from environments.collector.env import CollectorEnv
from environments.collector.params import EnvParams
from environments.collector.state import serialize_env_actions, serialize_env_states
from environments.collector.utils import to_numpy

class CollectorGymEnv(gym.Env):
    def __init__(self, numpy_output: bool = False):
        self.numpy_output = numpy_output
        self.rng_key = jax.random.key(0)
        self.jax_env = CollectorEnv(auto_reset=False)
        self.env_params: EnvParams = EnvParams()

        self.action_space = gym.spaces.Discrete(4)
    

    def reset(self, *, seed: int = None, options: dict[str, Any] | None = None) -> tuple[Any, Any]:
        if seed is not None:
            self.rng_key = jax.random.PRNGKey(seed)
        self.rng_key, reset_key = jax.random.split(self.rng_key)
        params = EnvParams()
        self.env_params = params
        obs, self.state = self.jax_env.reset(reset_key, params=params)
        if self.numpy_output:
            obs = to_numpy(flax.serialization.to_state_dict(obs))
        
        return obs, dict(params=params, state=self.state)
    
    def step(self, action: int) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        self.rng_key, step_key = jax.random.split(self.rng_key)
        obs, self.state, reward, terminated, truncated, info = self.jax_env.step(
            step_key, self.state, action, self.env_params
        )
        if self.numpy_output:
            obs = to_numpy(flax.serialization.to_state_dict(obs))
            reward = to_numpy(reward)
            terminated = to_numpy(terminated)
            truncated = to_numpy(truncated)
    
        return obs, reward, terminated, truncated, info


class RecordEpisode(gym.Wrapper):
    def __init__(
        self,
        env: CollectorGymEnv,
        save_dir: str = None,
        save_on_close: bool = True,
        save_on_reset: bool = True,
    ):
        super().__init__(env)
        self.episode = dict(states=[], actions=[], rewards=[], metadata=dict())
        self.episode_id = 0
        
        self.save_on_close = save_on_close
        self.save_on_reset = save_on_reset
        self.episode_steps = 0
        self.save_dir = save_dir
        if save_dir is not None:
            from pathlib import Path
            Path(save_dir).mkdir(parents=True, exist_ok=True)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        if self.save_on_reset and self.episode_steps > 0:
            self._save_episode_and_reset()
        obs, info = self.env.reset(seed=seed, options=options)
        

        self.episode["metadata"]["seed"] = seed
        #self.episode["params"] = flax.serialization.to_state_dict(info["full_params"])
        self.episode["states"].append(info["state"])
        return obs, info
    
    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1
        self.episode["states"].append(info["state"])
        self.episode["actions"].append(action)
        self.episode["rewards"].append(reward.tolist())
        return obs, reward, terminated, truncated, info

    def serialize_episode_data(self, episode=None):
        if episode is None:
            episode = self.episode
        ret = dict()
        ret["observations"] = serialize_env_states(episode["states"])
        if "actions" in episode:
            ret["actions"] = serialize_env_actions(episode["actions"])
        ret["metadata"] = episode["metadata"]
        ret["rewards"] = episode["rewards"]
        #ret["params"] = episode["params"]
        return ret

    def save_episode(self, save_path: str):
        episode = self.serialize_episode_data()
        with open(save_path, "w") as f:
            json.dump(episode, f)
        self.episode = dict(states=[], actions=[], rewards=[], metadata=dict())

    def _save_episode_and_reset(self):
        """saves to generated path based on self.save_dir and episoe id and updates relevant counters"""
        self.save_episode(
            os.path.join(self.save_dir, f"episode_{self.episode_id}.json")
        )
        self.episode_id += 1
        self.episode_steps = 0

    def close(self):
        if self.save_on_close and self.episode_steps > 0:
            self._save_episode_and_reset()







