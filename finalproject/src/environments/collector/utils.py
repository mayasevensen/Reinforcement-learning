import jax
import jax.numpy as jnp
import numpy as np
from jax import lax





def get_observable_region(state):
    """Sets observable tiles to 1 in a 9x9 region around the unit, 0 elsewhere."""
    grid_h, grid_w = state.map_features.tile_type.shape
    y, x = state.units.position  # jnp.array([y, x])
    view_radius = 4

    # Create coordinate grids
    row_idx = jnp.arange(grid_h).reshape(-1, 1)  # Shape (H, 1)
    col_idx = jnp.arange(grid_w).reshape(1, -1)  # Shape (1, W)

    # Compute observable mask using broadcasting
    in_y = jnp.abs(row_idx - y) <= view_radius
    in_x = jnp.abs(col_idx - x) <= view_radius

    observable_mask = jnp.logical_and(in_y, in_x).astype(jnp.int8)
    
    state = state.replace(observable=observable_mask)
    return state


def extract_observable_region(state):
    # 9x9 observable area around the unit
    unit_pos = state.units.position  # [y, x] position
    # Pad map to avoid out-of-bounds errors
    padded_tile_type = jnp.pad(state.map_features.tile_type, pad_width=4, mode="constant", constant_values=1)

    # Adjust unit position for padding
    adjusted_unit_pos = unit_pos + 4

    # Extract 9x9 observable region using `lax.dynamic_slice`
    observable_map = jax.lax.dynamic_slice(
        padded_tile_type,
        start_indices=(adjusted_unit_pos[0] - 4, adjusted_unit_pos[1] - 4),
        slice_sizes=(9, 9)
    )
    return observable_map


def to_numpy(x):
    if isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    elif isinstance(x, list):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)