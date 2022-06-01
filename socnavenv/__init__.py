from gym.envs.registration import register

register(
    id='SocNavEnv-v0',
    entry_point='socnavenv.envs:SocNavEnv',
)

register(
    id='SocNavEnv-v1',
    entry_point='socnavenv.envs:SocNavEnv_v1',
)