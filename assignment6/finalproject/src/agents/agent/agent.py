from agents.agent_base import BaseAgent
from types import SimpleNamespace
from environments.collector.state import EnvState

class Agent(BaseAgent):
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.config = config
        
    def load(self, path: str) -> None:
        raise NotImplementedError("Implement the act method")

    def act(self, observation: EnvState) -> int:
        raise NotImplementedError("Implement the act method")
        
        
