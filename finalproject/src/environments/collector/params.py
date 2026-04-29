from flax import struct
import jax
import chex
import jax.numpy as jnp

@struct.dataclass
class EnvParams():
    map_width: int = 16
    map_height: int = 16
    number_of_initial_items: int = 10
    
   

