"""
BFS Greedy Agent for the Collector environment.

Finds the true shortest path (via BFS) to the nearest reachable item,
navigating around obstacles. This is a near-optimal greedy collector
and makes a strong training opponent for DQN agents.
"""

from agents.agent_base import BaseAgent
from types import SimpleNamespace
from environments.collector.state import EnvState
from collections import deque
import numpy as np


class Agent(BaseAgent):
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.config = config
        self.epsilon = getattr(config, 'epsilon', 0.05)
        self.action_space = getattr(config, 'action_space', 4)
        self._path = []  # cached action path to current target
        self._target = None

    def load(self) -> None:
        pass

    # ------------------------------------------------------------------
    # BFS: returns the first action to take from `start` to reach `goal`
    # on the given tile_map (tile_type == 1 is obstacle).
    # Returns None if goal is unreachable.
    # ------------------------------------------------------------------
    @staticmethod
    def _bfs_first_action(tile_map, start, goal):
        """
        BFS from start to goal on tile_map.
        Returns the action index (0=UP,1=RIGHT,2=DOWN,3=LEFT) of the
        first step, or None if unreachable.
        """
        H, W = tile_map.shape
        sy, sx = int(start[0]), int(start[1])
        gy, gx = int(goal[0]),  int(goal[1])

        if (sy, sx) == (gy, gx):
            return None  # already there

        # directions: UP, RIGHT, DOWN, LEFT
        DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        # visited stores (parent_y, parent_x, action_taken_from_parent)
        visited = {}
        visited[(sy, sx)] = None
        queue = deque([(sy, sx)])

        while queue:
            cy, cx = queue.popleft()
            for action, (dy, dx) in enumerate(DIRS):
                ny, nx = cy + dy, cx + dx
                if not (0 <= ny < H and 0 <= nx < W):
                    continue
                if (ny, nx) in visited:
                    continue
                if tile_map[ny, nx] == 1:   # obstacle
                    continue
                visited[(ny, nx)] = (cy, cx, action)
                if (ny, nx) == (gy, gx):
                    # Trace back to find the FIRST action taken
                    node = (ny, nx)
                    while True:
                        py, px, act = visited[node]
                        if (py, px) == (sy, sx):
                            return act
                        node = (py, px)
                queue.append((ny, nx))

        return None  # unreachable

    # ------------------------------------------------------------------
    # Find the nearest reachable item and return BFS distance + position
    # ------------------------------------------------------------------
    @staticmethod
    def _nearest_reachable_item(tile_map, pos):
        """
        Full BFS from pos; returns (item_position, bfs_distance) for the
        nearest item (tile_type == 2), or (None, inf) if none exist.
        """
        H, W = tile_map.shape
        sy, sx = int(pos[0]), int(pos[1])

        dist = -np.ones((H, W), dtype=np.int32)
        dist[sy, sx] = 0
        queue = deque([(sy, sx)])
        DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        while queue:
            cy, cx = queue.popleft()
            if tile_map[cy, cx] == 2:   # found an item
                return (cy, cx), int(dist[cy, cx])
            for dy, dx in DIRS:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < H and 0 <= nx < W and dist[ny, nx] == -1 and tile_map[ny, nx] != 1:
                    dist[ny, nx] = dist[cy, cx] + 1
                    queue.append((ny, nx))

        return None, float('inf')   # no reachable items

    # ------------------------------------------------------------------
    # Act
    # ------------------------------------------------------------------
    def act(self, observation: EnvState) -> int:
        # Epsilon-random exploration (kept small so agent is near-optimal)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space)

        tile_map = observation["map_features"]["tile_type"]
        pos = observation["units"]["position"][0]   # (y, x)

        # Find nearest reachable item via BFS
        target, _ = self._nearest_reachable_item(tile_map, pos)

        if target is None:
            # No items on map — move randomly
            return np.random.randint(self.action_space)

        # Get first action toward target via BFS
        action = self._bfs_first_action(tile_map, pos, target)

        if action is None:
            # Already on item or unreachable — random fallback
            return np.random.randint(self.action_space)

        return action