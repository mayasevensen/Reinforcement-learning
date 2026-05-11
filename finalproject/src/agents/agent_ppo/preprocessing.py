"""
Observation preprocessing for the Collector environment.

The key trick: we precompute BFS distance maps from each player's position.
This gives the network true reachability (around obstacles) without it
having to learn pathfinding from raw pixels. With this, the network can
focus on the strategic layer: which item to go for given contention.

Output channels (8 total):
    0: walls            (1 where obstacle)
    1: items            (1 where item)
    2: my position      (1 at my cell)
    3: opponent position (1 at opponent cell)
    4: my BFS-dist to every cell, normalized to [0,1] (1.0 = unreachable)
    5: opponent BFS-dist, same encoding
    6: contention map: who is closer? (-1 = me, +1 = opp, 0 = tie/unreach)
    7: 1 / (1 + my_dist_to_nearest_item) heatmap centered on items I'd target

A 16x16 board with 8 channels = 2048 floats per obs. Cheap on CPU.
"""
from collections import deque
import numpy as np


# Indexed as (dy, dx): UP, RIGHT, DOWN, LEFT
DIRS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int8)
NUM_CHANNELS = 8
GRID_H = 16
GRID_W = 16


def bfs_distances(tile_map: np.ndarray, start_y: int, start_x: int) -> np.ndarray:
    """
    Compute BFS distance from (start_y, start_x) to every cell.
    Obstacle cells (tile == 1) are unreachable. Items (tile == 2) are walkable.
    Returns int array of shape (H, W); -1 means unreachable.
    """
    H, W = tile_map.shape
    dist = np.full((H, W), -1, dtype=np.int16)
    if not (0 <= start_y < H and 0 <= start_x < W):
        return dist
    if tile_map[start_y, start_x] == 1:
        # Shouldn't happen (player can't be on an obstacle) but guard anyway
        return dist

    dist[start_y, start_x] = 0
    q = deque()
    q.append((start_y, start_x))
    while q:
        y, x = q.popleft()
        d = dist[y, x] + 1
        for dy, dx in DIRS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and dist[ny, nx] == -1 and tile_map[ny, nx] != 1:
                dist[ny, nx] = d
                q.append((ny, nx))
    return dist


def encode_observation(obs: dict) -> np.ndarray:
    """
    Convert an environment observation dict into a (C, H, W) float32 tensor.
    Expects the player_X dict (already from that player's perspective).
    """
    tile_map = np.asarray(obs["map_features"]["tile_type"], dtype=np.int8)
    pos = np.asarray(obs["units"]["position"], dtype=np.int32)  # (2, 2): [me, opp]
    H, W = tile_map.shape

    my_y, my_x = int(pos[0, 0]), int(pos[0, 1])
    op_y, op_x = int(pos[1, 0]), int(pos[1, 1])

    # Clip in case of out-of-bounds (shouldn't happen but be safe)
    my_y = max(0, min(H - 1, my_y))
    my_x = max(0, min(W - 1, my_x))
    op_y = max(0, min(H - 1, op_y))
    op_x = max(0, min(W - 1, op_x))

    walls = (tile_map == 1).astype(np.float32)
    items = (tile_map == 2).astype(np.float32)

    me_pos = np.zeros((H, W), dtype=np.float32)
    me_pos[my_y, my_x] = 1.0
    op_pos = np.zeros((H, W), dtype=np.float32)
    op_pos[op_y, op_x] = 1.0

    my_dist_int = bfs_distances(tile_map, my_y, my_x)
    op_dist_int = bfs_distances(tile_map, op_y, op_x)

    # Normalize: map -1 (unreachable) -> 1.0 (max). Reachable cells in [0, ~1).
    norm = float(H + W)
    my_dist = np.where(my_dist_int < 0, 1.0, my_dist_int.astype(np.float32) / norm)
    op_dist = np.where(op_dist_int < 0, 1.0, op_dist_int.astype(np.float32) / norm)

    # Contention map: -1 if I'm closer, +1 if opp closer, 0 if tie / both unreachable
    both_reach = (my_dist_int >= 0) & (op_dist_int >= 0)
    me_closer = both_reach & (my_dist_int < op_dist_int)
    op_closer = both_reach & (op_dist_int < my_dist_int)
    contention = np.zeros((H, W), dtype=np.float32)
    contention[me_closer] = -1.0
    contention[op_closer] = 1.0

    # "Item attractor": for each item, value = 1/(1 + my_dist) if reachable
    item_attract = np.zeros((H, W), dtype=np.float32)
    item_mask = (tile_map == 2)
    if item_mask.any():
        reachable_items = item_mask & (my_dist_int >= 0)
        item_attract[reachable_items] = 1.0 / (1.0 + my_dist_int[reachable_items].astype(np.float32))

    out = np.stack([
        walls,        # 0
        items,        # 1
        me_pos,       # 2
        op_pos,       # 3
        my_dist,      # 4
        op_dist,      # 5
        contention,   # 6
        item_attract, # 7
    ], axis=0)
    return out  # (8, 16, 16) float32


def shaping_potential(obs: dict) -> float:
    """
    Potential for potential-based reward shaping (Ng, Harada, Russell 1999).
    F(s, s') = gamma * phi(s') - phi(s) leaves the optimal policy invariant
    for ANY horizon - perfect for variable-length episodes.

    We use phi(s) = -alpha * BFS_dist_to_nearest_uncontested_item.
    "Uncontested" = items where I'm strictly closer than the opponent.
    Falls back to nearest reachable item if no uncontested ones.
    """
    tile_map = np.asarray(obs["map_features"]["tile_type"], dtype=np.int8)
    pos = np.asarray(obs["units"]["position"], dtype=np.int32)
    my_y, my_x = int(pos[0, 0]), int(pos[0, 1])
    op_y, op_x = int(pos[1, 0]), int(pos[1, 1])
    H, W = tile_map.shape
    my_y = max(0, min(H - 1, my_y)); my_x = max(0, min(W - 1, my_x))
    op_y = max(0, min(H - 1, op_y)); op_x = max(0, min(W - 1, op_x))

    item_mask = (tile_map == 2)
    if not item_mask.any():
        return 0.0

    my_dist = bfs_distances(tile_map, my_y, my_x)
    op_dist = bfs_distances(tile_map, op_y, op_x)

    item_my = my_dist[item_mask]
    item_op = op_dist[item_mask]

    # Uncontested: reachable to me AND I'm strictly closer than opponent
    # (or opponent can't reach it at all)
    reachable_me = item_my >= 0
    op_unreach = item_op < 0
    me_closer = reachable_me & (op_unreach | (item_my < item_op))

    if me_closer.any():
        d = int(item_my[me_closer].min())
    elif reachable_me.any():
        d = int(item_my[reachable_me].min())
    else:
        return 0.0

    return -float(d)