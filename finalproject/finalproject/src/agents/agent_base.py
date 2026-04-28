from abc import ABC, abstractmethod
from typing import Any, Dict, Union
from environments.collector.state import EnvState
from types import SimpleNamespace
class BaseAgent(ABC):
    def __init__(self,config: SimpleNamespace):
        """
        Initialize the agent with a configuration dictionary.
        """
        self.config = config

    @abstractmethod
    def act(self, observation: EnvState) -> int:
        """
        Given an observation, return an action.
        The action is an integer.
        """
        pass

    @abstractmethod
    def load(self) -> None:
        """
        Load model parameters or state. use save_path from config.yaml.
        """
        pass