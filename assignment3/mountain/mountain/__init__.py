from gymnasium.envs.registration import register


register(
    id='mountain/GridWorld-v1',
    entry_point="mountain.envs:MountainEnv",
    kwargs={"backwards": False},  # Specify version 1
)

register(
    id='mountain/GridWorld-v2',
    entry_point="mountain.envs:MountainEnv",
    kwargs={"backwards": True},  # Specify version 1
)