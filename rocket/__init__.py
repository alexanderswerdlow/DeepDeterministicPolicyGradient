from gym.envs.registration import register

register(
    id='InvertedPendulum-v3',
    entry_point='rocket.envs:InvertedPendulumEnv',
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    id='RocketEnv-v1',
    entry_point='rocket.envs:QuadRateEnv',
    max_episode_steps=8000,
    reward_threshold=9600.0,
)

register(
    id='RocketEnv-v2',
    entry_point='rocket.envs:RocketV2',
    max_episode_steps=8000,
    reward_threshold=9600.0,
)