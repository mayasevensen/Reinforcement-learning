from gymnasium.envs.registration import register

register(
    id='cookiedisaster',
    entry_point="cookiedisaster.envs:CookieDisasterEnv",
)