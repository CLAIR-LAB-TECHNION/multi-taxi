import copy

import pytest

from multi_taxi import multi_taxi_v0, single_taxi_v0


@pytest.mark.parametrize(['make_env', 'actions'], [(multi_taxi_v0.parallel_env, {'taxi_0': 0}),
                                                   (single_taxi_v0.gym_env, 0)])
def test_deepcopy(make_env, actions):
    env = make_env(num_taxis=1)
    copy.deepcopy(env).reset(seed=0)  # copy before reset

    env.reset(seed=0)
    env_copy = copy.deepcopy(env)
    state = env.unwrapped.state()
    env_copy.step(actions)

    assert env_copy.unwrapped.state() != state
    assert env.unwrapped.state() == state
