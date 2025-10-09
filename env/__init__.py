from gymnasium.envs.registration import register

register(
    id='heal-cube-single-v0',
    entry_point='envs.heal:CubeEnv',
    max_episode_steps=200,
    kwargs=dict(
        env_type='single',
    ),
)