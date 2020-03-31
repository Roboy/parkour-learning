import gym
from gym.envs.registration import registry, make, spec


def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kvargs)


register(
    id='HumanoidPrimitivePretraining-v0',
    entry_point='parkour_learning.gym_env.primitive_pretraining_env:PrimitivePretrainingEnv',
)

register(
    id='TrackEnv-v0',
    entry_point='parkour_learning.gym_env.track_env:TrackEnv',
)
