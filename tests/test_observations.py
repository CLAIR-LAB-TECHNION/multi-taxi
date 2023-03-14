import pytest
from gymnasium.spaces import MultiDiscrete, Box, Dict

from multi_taxi import multi_taxi_v0, ObservationType
from .common import test_env_cfgs

TRUE_SPACES = {
    'single_taxi_simple': {'taxi_0': [7, 12, 7, 12, 7, 12, 2, 2]},
    'single_taxi_two_passenger': {'taxi_0': [7, 12, 7, 12, 7, 12, 2, 2, 7, 12, 7, 12, 2, 2]},
    'single_taxi_three_passenger': {'taxi_0': [7, 12, 7, 12, 7, 12, 2, 2, 7, 12, 7, 12, 2, 2, 7, 12, 7, 12, 2, 2]},
    'single_taxi_four_passenger': {
        'taxi_0': [7, 12, 7, 12, 7, 12, 2, 2, 7, 12, 7, 12, 2, 2, 7, 12, 7, 12, 2, 2, 7, 12, 7, 12, 2, 2]
    },
    'single_taxi_five_passenger': {
        'taxi_0': [7, 12, 7, 12, 7, 12, 2, 2, 7, 12, 7, 12, 2, 2, 7, 12, 7, 12, 2, 2, 7, 12, 7, 12, 2, 2, 7, 12, 7, 12,
                   2, 2]
    },
    'single_taxi_two_passenger_pickup_only': {'taxi_0': [7, 12, 7, 12, 2, 7, 12, 2]},
    'single_taxi_three_passenger_pickup_only': {'taxi_0': [7, 12, 7, 12, 2, 7, 12, 2, 7, 12, 2]},
    'single_taxi_four_passenger_pickup_only': {'taxi_0': [7, 12, 7, 12, 2, 7, 12, 2, 7, 12, 2, 7, 12, 2]},
    'single_taxi_five_passenger_pickup_only': {'taxi_0': [7, 12, 7, 12, 2, 7, 12, 2, 7, 12, 2, 7, 12, 2, 7, 12, 2]},
    'single_taxi_fuel': {'taxi_0': [7, 12, 101, 2, 7, 12, 7, 12, 2, 2]},
    'single_taxi_engine_control': {'taxi_0': [7, 12, 2, 7, 12, 7, 12, 2, 2]},
    'single_taxi_specify_pickup': {'taxi_0': [7, 12, 7, 12, 7, 12, 2, 2, 7, 12, 7, 12, 2, 2, 7, 12, 7, 12, 2, 2]},
    'single_taxi_specify_dropoff': {'taxi_0': [7, 12, 7, 12, 7, 12, 2, 2, 7, 12, 7, 12, 2, 2, 7, 12, 7, 12, 2, 2]},
    'single_taxi_specify_pickup_and_dropoff': {
        'taxi_0': [7, 12, 7, 12, 7, 12, 2, 2, 7, 12, 7, 12, 2, 2, 7, 12, 7, 12, 2, 2]
    },
    'single_taxi_small_fuel_map': {'taxi_0': [4, 4, 4, 4, 4, 4, 2, 2]},
    'single_taxi_simple_small_map_no_obs': {'taxi_0': [5, 5, 5, 5, 5, 5, 2, 2]},
    'two_taxi_simple': {f'taxi_{i}': [7, 12, 7, 12, 7, 12, 2, 2, 2] for i in range(2)},
    'three_taxi_simple': {f'taxi_{i}': [7, 12, 7, 12, 7, 12, 2, 2, 2, 2] for i in range(3)},
    'four_taxi_simple': {f'taxi_{i}': [7, 12, 7, 12, 7, 12, 2, 2, 2, 2, 2] for i in range(4)},
    'five_taxi_simple': {f'taxi_{i}': [7, 12, 7, 12, 7, 12, 2, 2, 2, 2, 2, 2] for i in range(5)},
    'two_taxi_see_each_other': {f'taxi_{i}': [7, 12, 7, 12, 7, 12, 7, 12, 2, 2, 2] for i in range(2)},
    'three_taxi_see_each_other': {f'taxi_{i}': [7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 2, 2, 2, 2] for i in range(3)},
    'four_taxi_see_each_other': {f'taxi_{i}': [7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 2, 2, 2, 2, 2]
                                 for i in range(4)},
    'five_taxi_see_each_other': {f'taxi_{i}': [7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 2, 2, 2, 2, 2, 2]
                                 for i in range(5)},
    'two_taxi_fuel_limit': {
        'taxi_0': [7, 12, 11, 2, 7, 12, 7, 12, 2, 2, 2],
        'taxi_1': [7, 12, 6, 2, 7, 12, 7, 12, 2, 2, 2],
    },
    'three_taxi_fuel_limit': {
        'taxi_0': [7, 12, 16, 2, 7, 12, 7, 12, 2, 2, 2, 2],
        'taxi_1': [7, 12, 26, 2, 7, 12, 7, 12, 2, 2, 2, 2],
        'taxi_2': [7, 12, 100, 2, 7, 12, 7, 12, 2, 2, 2, 2]
    },
    'two_taxi_small_map': {f'taxi_{i}': [5, 5, 5, 5, 5, 5, 2, 2, 2] for i in range(2)},
    'two_taxi_see_each_other_small_map': {f'taxi_{i}': [5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2] for i in range(2)},
    'single_taxi_image_observation': {'taxi_0': [0, 255, (140, 226, 3)]},
    'single_taxi_multi_observation': {'taxi_0': {
            'symbolic': [7, 12, 7, 12, 7, 12, 2, 2, 7, 12, 7, 12, 2, 2],
            'image': [0, 255, (140, 226, 3)]
    }},
    'three_taxi_multi_observation': {
        'taxi_0': [7, 12, 31, 2, 7, 12, 7, 12, 2, 2, 2, 2, 7, 12, 7, 12, 2, 2, 2, 2],
        'taxi_1': [0, 255, (140, 226, 3)],
        'taxi_2': {
            'symbolic': [7, 12, 7, 12, 7, 12, 2, 2, 2, 2, 7, 12, 7, 12, 2, 2, 2, 2],
            'image': [0, 255, (140, 226, 3)]
        }
    }
}


@pytest.mark.parametrize(['env_name', 'env_cfg'], test_env_cfgs.items())
def test_observation_spaces(env_name, env_cfg):
    assert env_name in TRUE_SPACES, f'missing ground-truth observation space for env: {env_name}'

    check_observation_space_validity(env_cfg, TRUE_SPACES[env_name])


def check_observation_space_validity(taxi_env_params, expected_observation_space):
    env = multi_taxi_v0.parallel_env(**taxi_env_params)
    obs = env.reset()

    # make sure expected observations have the same keys as the true observations
    assert set(obs.keys()) == set(expected_observation_space.keys())

    # check correct observation space and observations
    for agent, agent_obs in obs.items():
        obs_type = env.unwrapped.observation_type[agent]
        obs_space = env.observation_space(agent)

        if obs_type == ObservationType.SYMBOLIC:
            assert MultiDiscrete(expected_observation_space[agent]) == obs_space
            assert len(env.unwrapped.get_observation_meanings(agent)) == len(obs_space)
        elif obs_type == ObservationType.IMAGE:
            assert Box(*expected_observation_space[agent]) == obs_space
        else:
            sym_obs = expected_observation_space[agent][ObservationType.SYMBOLIC.value]
            img_obs = expected_observation_space[agent][ObservationType.IMAGE.value]
            assert Dict(spaces={
                ObservationType.SYMBOLIC.value: MultiDiscrete(sym_obs),
                ObservationType.IMAGE.value: Box(*img_obs)
            }) == obs_space

            # check symbolic observation space length same as meaning length:
            meanings_length = len(env.get_observation_meanings(agent)[ObservationType.SYMBOLIC.value])
            obs_space_length = len(obs_space.spaces[ObservationType.SYMBOLIC.value])
            assert meanings_length == obs_space_length

        assert obs_space.contains(agent_obs)
