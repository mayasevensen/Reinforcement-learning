from agents.agent_base import BaseAgent
from types import SimpleNamespace
from environments.collector.state import EnvState
from collections import deque
import numpy as np
class Agent(BaseAgent):
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.config = config
        self.seed = config.seed
        self.action_space = config.action_space
        np.random.seed(self.seed)
        
    def load(self) -> None:
        pass

    def act(self, observation: EnvState) -> int:
        return np.random.choice(self.action_space)
        
        

    
        
        
