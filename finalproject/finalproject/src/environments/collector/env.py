import functools
from typing import Any, Dict, Optional, Tuple, Union
import chex
import gymnax
import jax
import jax.numpy as jnp
from jax import lax

import numpy as np

from gymnax.environments import environment, spaces
from environments.collector.params import EnvParams
from environments.collector.state import (
    EMPTY_TILE,
    OBSTACLE_TILE,
    ITEM_TILE,
    EnvState,
    MapTile,
    UnitState,
    gen_state
)

class CollectorEnv(environment.Environment):
    def __init__(
        self, auto_reset=False, fixed_env_params: EnvParams = EnvParams(),
          **kwargs
    ):
        super().__init__(**kwargs)
        self.auto_reset = auto_reset
        self.fixed_env_params = fixed_env_params
    
    @property
    def default_params(self) -> EnvParams:
        params = EnvParams()
        params = jax.jax.tree.map(jax.numpy.array, params)
        return params
    

    def get_obs(self, state: EnvState):
        obs = {}
        obs["player_0"] = EnvState(
            units=state.units,
            map_features=state.map_features,
            team_points=state.team_points,
            items_on_map=state.items_on_map,
            steps=state.steps
        )
        def flip_state_for_player1(state: EnvState) -> EnvState:
            tile_type = state.map_features.tile_type
            height, width = tile_type.shape

            # Flip map_features along anti-diagonal
            flipped_tile_type = jnp.transpose(tile_type)[::-1, ::-1]

            # Flip and swap unit positions
            def flip_position(pos):
                y, x = pos
                return jnp.array([width - 1 - x, height - 1 - y], dtype=jnp.int16)

            flipped_positions = jax.vmap(flip_position)(state.units.position)
            flipped_positions = flipped_positions[jnp.array([1, 0])]  # Swap player 0 and player 1

            # Swap team points
            flipped_team_points = state.team_points[jnp.array([1, 0])]

            return EnvState(
                units=UnitState(position=flipped_positions),
                map_features=MapTile(tile_type=flipped_tile_type),
                team_points=flipped_team_points,
                items_on_map=state.items_on_map,
                steps=state.steps
            )
        obs["player_1"] = flip_state_for_player1(state)
        return obs
        
        





    #@functools.partial(jax.jit, static_argnums=(0, 4))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int,int],
        params: EnvParams,
    ) -> Tuple[EnvState, float, bool, bool]:
        prev_team_points = state.team_points

        # 1 is move up, 2 is move right, 3 is move down, 4 is move left
        # Define movement directions
        directions = jnp.array(
            [
                [-1, 0],  # Move up
                [0, 1],  # Move right
                [1, 0],  # Move down
                [0, -1],  # Move left
            ],
            dtype=jnp.int16,
        )
        

        def move_unit(unit: UnitState, action):
            new_position = unit.position + directions[action]
            is_blocked = (
                state.map_features.tile_type[new_position[0], new_position[1]] == OBSTACLE_TILE
            )
            hit_wall = jnp.any(new_position < 0) | jnp.any(new_position >= jnp.array([params.map_width, params.map_height]))
            
            
            unit_moved = (
                ~is_blocked & ~hit_wall
            )
            
            return (UnitState(position=jnp.where(unit_moved, new_position, unit.position)), unit_moved)

        # Move units
        flip_action_map = jnp.array([1, 0, 3, 2])
        action_p0 = action["player_0"]
        action_p1 = flip_action_map[action["player_1"]] # both agent see the environment as if they are player 0. So we need to flip the action of player 1
        actions = jnp.stack([action_p0, action_p1])
        moved_units, moved = jax.vmap(move_unit)(state.units, actions)
        
        state = state.replace(units=moved_units)
    

        def handle_item_collection(state: EnvState) -> EnvState:
            """Handles item collection for multiple agents."""
            tile_type = state.map_features.tile_type
            positions = state.units.position  # shape (2, 2): [team, (y, x)]

            def is_item(pos):
                y, x = pos
                return tile_type[y, x] == ITEM_TILE

            is_on_item = jax.vmap(is_item)(positions)  # shape (2,)
            updated_team_points = state.team_points + is_on_item.astype(jnp.int16)

            # Get positions to clear
            ys, xs = positions[:, 0], positions[:, 1]

            # Only update positions where items exist
            def clear_if_item(tile_map, y, x, mask):
                return jax.lax.cond(
                    mask,
                    lambda t: t.at[y, x].set(EMPTY_TILE),
                    lambda t: t,
                    tile_map
                )

            updated_tile_type = tile_type
            for i in range(2):
                updated_tile_type = clear_if_item(updated_tile_type, ys[i], xs[i], is_on_item[i])

            items_on_map = jnp.expand_dims(jnp.sum(updated_tile_type == ITEM_TILE), axis=0)
    

            return state.replace(
                map_features=state.map_features.replace(tile_type=updated_tile_type),
                team_points=updated_team_points,
                items_on_map=items_on_map
            )



        state = handle_item_collection(state)
        # Update state's step count
        state = state.replace(steps=state.steps + 1)
        
        def maybe_spawn_item(state: EnvState, key: chex.PRNGKey) -> EnvState:
            """Spawns a new item at a random walkable and unoccupied tile every 100 steps."""
            # Ratio of remaining items
            item_ratio = state.items_on_map / params.number_of_initial_items

            # Dynamically adjust spawn_step: fewer items → smaller step (faster spawn)
            # Clamp to avoid divide-by-zero and keep spawn_step in a reasonable range
            min_step = 1
            max_step = 25
            spawn_step = jnp.clip((item_ratio * max_step).astype(jnp.int32), min_step, max_step)

            should_spawn = jnp.logical_and(state.steps % spawn_step == 0, state.items_on_map < (params.number_of_initial_items-1)).squeeze()
            

            
            def spawn_fn(s):
                tile_type = s.map_features.tile_type
                height, width = tile_type.shape
                agent_y, agent_x = s.units.position

                # Create valid spawn mask (must be empty, not the agent position, and its mirror must also be empty)
                valid_mask = (tile_type == EMPTY_TILE).astype(jnp.int32)
                valid_mask = valid_mask.at[agent_y, agent_x].set(0)

                # Create mirror mask: (y, x) -> (width - 1 - x, height - 1 - y)
                y_coords, x_coords = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
                mirrored_y = width - 1 - x_coords
                mirrored_x = height - 1 - y_coords

                mirrored_is_empty = tile_type[mirrored_y, mirrored_x] == EMPTY_TILE
                valid_mask = valid_mask * mirrored_is_empty.astype(jnp.int32)

                # Flatten mask and get valid indices
                flat_mask = valid_mask.ravel()
                possible_indices = jnp.nonzero(flat_mask, size=flat_mask.size)[0]
                num_valid = jnp.sum(flat_mask)

                def spawn_at_random(s):
                    spawn_idx = jax.random.choice(key, possible_indices, shape=(), replace=False)
                    y, x = jnp.unravel_index(spawn_idx, (height, width))
                    y_mirror = width - 1 - x
                    x_mirror = height - 1 - y

                    new_tile_type = tile_type.at[y, x].set(ITEM_TILE)
                    new_tile_type = new_tile_type.at[y_mirror, x_mirror].set(ITEM_TILE)
                    new_map = s.map_features.replace(tile_type=new_tile_type)
                    new_items = s.items_on_map + 2

                    return s.replace(map_features=new_map, items_on_map=new_items)

                return jax.lax.cond(num_valid > 0, spawn_at_random, lambda s: s, s)

            # Only spawn if `should_spawn` is True
            return jax.lax.cond(should_spawn, spawn_fn, lambda s: s, state)

        state = maybe_spawn_item(state, key)
        delta_team_points = state.team_points - prev_team_points
        reward = delta_team_points - 1*(2-moved.astype(jnp.int32)) # -2 if crashed,-1 if no item collected, 0 if item collected
        terminated = jnp.any(state.steps >= 1000)
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            terminated,
            False
        )
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: Optional[EnvParams] = None,
    ) -> Tuple[EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Performs step transitions in the environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, terminated, truncated = self.step_env(
            key, state, action, params
        )
        info = {}
        info["state"] = state_st
        info["observation"] = obs_st
        
        obs = obs_st
        state = state_st
        return obs, state, reward, terminated, truncated, info
        
    
    
    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[EnvState]:
        """Reset environment state by sampling initial position."""
        state = gen_state(
            key=key,
            map_width=self.fixed_env_params.map_width,
            map_height=self.fixed_env_params.map_height,
            max_items=self.fixed_env_params.number_of_initial_items,
        )


        #spawn units
        map_h = self.fixed_env_params.map_height
        map_w = self.fixed_env_params.map_width 
        unit_positions = jnp.array([
                                    [0, 0],                # team 0
                                    [map_h - 1, map_w - 1] # team 1
                                    ], dtype=jnp.int16)

        state = state.replace(units=UnitState(position=unit_positions))

        return self.get_obs(state), state
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[EnvParams] = None
    ) -> Tuple[EnvState]:
        """Performs resetting of environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params

        obs, state = self.reset_env(key, params)
        return obs, state   
    

    def action_space(self, params: Optional[EnvParams] = None):
        """Action space of the environment."""
        return spaces.Discrete(4)

