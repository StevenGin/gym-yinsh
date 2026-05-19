from gymnasium.envs.registration import register

register(
    id="yinsh-v0",
    entry_point="gym_yinsh.envs:YinshEnv",
)
