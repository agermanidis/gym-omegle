from gym.envs.registration import register


register(
    id="omegle-v0",
    entry_point="gym_omegle.envs:OmegleEnv",
    timestep_limit=50)
