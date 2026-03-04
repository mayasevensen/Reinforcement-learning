from gymnasium.envs.registration import register

register(
    id='skyscraper/GridWorld-v0',
    entry_point="skyscraper.envs:SkyscraperEnv",
)