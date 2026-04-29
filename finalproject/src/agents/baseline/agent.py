from agents.agent_base import BaseAgent
from types import SimpleNamespace
from environments.collector.state import EnvState
from collections import deque
import numpy as np
class Agent(BaseAgent):
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.config = config
        self.epsilon = config.epsilon
        self.seed = config.seed
        self.action_space = config.action_space
        np.random.seed(self.seed)
        self.current_target = None
        self.path_to_follow = []  # Store path to follow
        self.memory = deque(maxlen=10)
        
    def load(self) -> None:
        pass

    def act(self, observation: EnvState) -> int:
        tile_map = observation["map_features"]["tile_type"]
        pos = tuple(observation["units"]["position"][0])  # (y, x)
        self.memory.append(pos)
        if self.epsilon > np.random.rand():
            # # Directions: [UP, RIGHT, DOWN, LEFT]
            # directions = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)])
            # candidates = []
            # for i, (dy, dx) in enumerate(directions):
            #     ny, nx = pos[0] + dy, pos[1] + dx
            #     if 0 <= ny < tile_map.shape[0] and 0 <= nx < tile_map.shape[1]:
            #         if tile_map[ny, nx] != 1 and (ny, nx) not in self.memory:
            #             candidates.append(i)
            
            # if candidates:
            #     return np.random.choice(candidates)
            #else:
            return np.random.randint(self.action_space)  # fallback to any action
        # Find all item positions
        item_locs = np.argwhere(tile_map == 2)  # shape (N, 2)
        if item_locs.shape[0] == 0:
            return np.random.randint(self.action_space)

        # Compute Manhattan distances to all items
        pos_arr = np.array(pos)  # shape (2,)
        dists = np.abs(item_locs - pos_arr).sum(axis=1)  # shape (N,)
        closest_item = item_locs[np.argmin(dists)]  # shape (2,)

        # All directions
        directions = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=int)  # UP, RIGHT, DOWN, LEFT
        next_positions = directions + np.array(pos)  # shape (4, 2)
        next_positions = np.array(next_positions, dtype=int)

        # Bounds check
        H, W = tile_map.shape
        valid_mask = (
            (next_positions[:, 0] >= 0) &
            (next_positions[:, 0] < H) &
            (next_positions[:, 1] >= 0) &
            (next_positions[:, 1] < W)
        )

        # Obstacle check
        valid_next_positions = next_positions[valid_mask]
        tile_vals = tile_map[valid_next_positions[:, 0], valid_next_positions[:, 1]]
        not_obstacle = np.zeros(4, dtype=bool)
        not_obstacle[valid_mask] = tile_vals != 1

        # Memory check
        memory_mask = np.array([tuple(p) not in self.memory for p in next_positions])

        combined_mask = valid_mask & not_obstacle & memory_mask

        # Compute distances from candidate next positions to closest item
        manhattan_to_item = np.abs(next_positions - closest_item).sum(axis=1)  # shape (4,)

        # Set distances for invalid moves to large number
        manhattan_to_item[~combined_mask] = H + W + 1

        if np.all(~combined_mask):
            return np.random.randint(self.action_space)  # no good moves, explore randomly

        return np.argmin(manhattan_to_item)

        
        

   



        
        
