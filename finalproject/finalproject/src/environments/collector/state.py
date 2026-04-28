import functools
import chex
import flax
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from flax import struct
from environments.collector.params import EnvParams



EMPTY_TILE = 0
OBSTACLE_TILE = 1
ITEM_TILE = 2

@struct.dataclass
class UnitState:
    position: chex.Array


@struct.dataclass
class MapTile:
    tile_type: int
    


@struct.dataclass
class EnvState:
    units: UnitState         # units in the game with shape (T,2)
    map_features: MapTile    # map features with shape (H,W)
    team_points: chex.Array  # points of the team with shape (T,)
    items_on_map: chex.Array # items on the map
    steps: int               # number of steps taken in the game
    
    
   
    
@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def gen_map(
    key: chex.PRNGKey, map_height: int, map_width: int,
    max_items: int
) -> chex.Array:
    """
    Generates an initial game map state without relic nodes.

    Parameters:
        key (chex.PRNGKey): Random seed.
        params (EnvParams): Environment parameters.
        map_type (int): Type of the map.
        map_height (int): Height of the map.
        map_width (int): Width of the map.
       
    Returns:
        dict: A dictionary containing map features and energy node data.
    """
    

    
    # Initialize map tiles
    map_features = MapTile(
        tile_type=jnp.zeros((map_height, map_width), dtype=jnp.int16)
    )


    ### Generate obstacles tiles ###
    key, subkey = jax.random.split(key)
    perlin_noise = generate_perlin_noise_2d(subkey, (map_height, map_width), (8, 8),interpolant=interpolant_path)
    # Mirror along diagonal
    perlin_noise= jnp.fliplr(jnp.triu(perlin_noise) + jnp.triu(perlin_noise, 1).T )
    obstacles = (perlin_noise - perlin_noise.min()) / (perlin_noise.max() - perlin_noise.min())  # Normalize
    obstacles = jnp.where(obstacles < 0.6, 0, 1).astype(jnp.int16)  # Threshold to get obstacles 
    
    # Set first and last row to walkable (0)
    obstacles = obstacles.at[0, :].set(0)
    obstacles = obstacles.at[-1, :].set(0)
    
    # Set first and last column to walkable (0)
    obstacles = obstacles.at[:, 0].set(0)
    obstacles = obstacles.at[:, -1].set(0)

    map_features = map_features.replace(tile_type=jnp.where(obstacles, OBSTACLE_TILE, 0))
    
    # remove_isolated_diagonal_cells
    map_features = map_features.replace(tile_type = remove_isolated_diagonal_cells(map_features.tile_type, map_height, map_width))
    # generate items 

    key, subkey = jax.random.split(key)
    perlin_noise = generate_perlin_noise_2d(subkey, (map_height, map_width), (8, 8))

    # Keep only upper triangle of noise matrix
    upper_noise = jnp.triu(perlin_noise)
    noise = jnp.where(obstacles == 1, 0, upper_noise)

    # Get top k indices (half the items, rest will be mirrored)
    k = max_items // 2
    flat_indices = jnp.argsort(noise.ravel())[-k:]
    top_positions = jnp.column_stack(jnp.unravel_index(flat_indices, noise.shape))  # (k, 2)

    # Mirror positions along anti-diagonal: (i, j) → (width-1-j, height-1-i)
    mirrored_positions = jnp.stack([map_width - 1 - top_positions[:, 1],
                                    map_height - 1 - top_positions[:, 0]], axis=1)

    # Combine original + mirrored item positions
    all_item_positions = jnp.concatenate([top_positions, mirrored_positions], axis=0)

    # Place items in map
    updated_tile_type = map_features.tile_type.at[
        all_item_positions[:, 0], all_item_positions[:, 1]
    ].set(ITEM_TILE)

    # Update map
    map_features = map_features.replace(tile_type=updated_tile_type)

    return dict(
        map_features=map_features
    )

def check_isolated(tile_map,map_height,map_width, y, x):
    is_walkable = tile_map[y, x] == 0

    def get(y, x, fallback):
        return jnp.where(
            (0 <= y) & (y < map_height) & (0 <= x) & (x < map_width),
            tile_map[y, x],
            fallback,
        )

    up    = get(y - 1, x, OBSTACLE_TILE)
    down  = get(y + 1, x, OBSTACLE_TILE)
    left  = get(y, x - 1, OBSTACLE_TILE)
    right = get(y, x + 1, OBSTACLE_TILE)

    all_blocked = (up == OBSTACLE_TILE) & (down == OBSTACLE_TILE) & (left == OBSTACLE_TILE) & (right == OBSTACLE_TILE)
    return jnp.where(is_walkable & all_blocked, OBSTACLE_TILE, tile_map[y, x])

def remove_isolated_diagonal_cells(tile_map, map_height, map_width):
    def row_fn(y):
        return jax.vmap(lambda x: check_isolated(tile_map, map_height, map_width, y, x))(jnp.arange(map_width))

    updated_map = jax.vmap(row_fn)(jnp.arange(map_height))
    return updated_map


def serialize_env_states(env_states: list[EnvState]):
    def convert_numpy(obj):
        """Convert NumPy/JAX types to Python-native types."""
        if isinstance(obj, (np.ndarray, jnp.ndarray, chex.Array)):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        return obj
    def serialize_array(root: EnvState, arr, key_path: str = ""):
        if key_path in ["sensor_mask"]:
            return None  # Skip unwanted keys
        if isinstance(arr, dict):
            return {k: serialize_array(root, v, key_path + "/" + k if key_path else k) for k, v in arr.items()}
        return convert_numpy(arr)
    steps = []
    for state in env_states:
        state_dict = flax.serialization.to_state_dict(state)
        steps.append(serialize_array(state, state_dict))

    return steps


def serialize_env_actions(env_actions: list):
    def convert_numpy(obj):
        """Convert NumPy/JAX types to Python-native types."""
        if isinstance(obj, (np.ndarray, jnp.ndarray)):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        return obj

    def serialize_array(arr, key_path: str = ""):
        if isinstance(arr, dict):
            return {k: serialize_array(v, key_path + "/" + k if key_path else k) for k, v in arr.items()}
        return convert_numpy(arr)

    steps = []
    for action in env_actions:
        action_dict = flax.serialization.to_state_dict(action)
        steps.append(serialize_array(action_dict))

    return steps


@functools.partial(jax.jit, static_argnums=( 1, 2, 3))
def gen_state(
    key: chex.PRNGKey,map_width: int, map_height: int,
    max_items: int 
) -> EnvState:
    """
    Generates an initial `EnvState` based on environment parameters.
    """
   

    generated = gen_map(key, map_width, map_height,max_items )
    # Initialize EnvState with correct attributes
    state = EnvState(
        units=UnitState(
            position=jnp.zeros((2, 2), dtype=jnp.int16), 
        ),
        
        map_features=generated["map_features"],
        team_points=jnp.zeros((2,), dtype=jnp.int16),
        items_on_map=jnp.zeros((1,), dtype=jnp.int16),
        steps=0
    )
    return state


def interpolant(t):
    return t*t*t*(t*(t*6 - 15) + 10)

def interpolant_cubic(t):
    """Improved cubic interpolant for smoother transitions."""
    return t * t * (3.0 - 2.0 * t)  # Cubic smoothstep

def interpolant_path(t):
    """Path-like interpolant by adding a directional bias."""
    return 0.5 * (1 - jnp.cos(t * jnp.pi)) + 0.2 * t**2  # Adds a linear component

@functools.partial(jax.jit, static_argnums=(1, 2, 3, 4))
def generate_perlin_noise_2d(
    key, shape, res, tileable=(False, False), interpolant=interpolant
):
    """Generate a 2D numpy array of perlin noise.

    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of shape shape with the generated noise.

    Raises:
        ValueError: If shape is not a multiple of res.
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = jnp.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]]\
             .transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*jnp.pi*jax.random.uniform(key, (res[0]+1, res[1]+1))
    gradients = jnp.dstack((jnp.cos(angles), jnp.sin(angles)))
    if tileable[0]:
        gradients[-1,:] = gradients[0,:]
    if tileable[1]:
        gradients[:,-1] = gradients[:,0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[    :-d[0],    :-d[1]]
    g10 = gradients[d[0]:     ,    :-d[1]]
    g01 = gradients[    :-d[0],d[1]:     ]
    g11 = gradients[d[0]:     ,d[1]:     ]
    
    # Ramps
    n00 = jnp.sum(jnp.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
    n10 = jnp.sum(jnp.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
    n01 = jnp.sum(jnp.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
    n11 = jnp.sum(jnp.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = interpolant(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return jnp.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)



