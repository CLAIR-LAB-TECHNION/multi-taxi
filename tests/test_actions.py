import pytest
from gymnasium.spaces import Discrete

from multi_taxi import multi_taxi_v0, Action
from multi_taxi.utils.types import MOVE_ACTIONS, ENGINE_ACTIONS
from .common import test_env_cfgs

MOVE_STRINGS = [a.value for a in MOVE_ACTIONS]
ENGINE_STRS = [a.value for a in ENGINE_ACTIONS]


def __actions_dict(lst):
    return {i: v for i, v in enumerate(lst)}


def __pickup_specify(num_passengers):
    return __actions_dict(MOVE_STRINGS +
                          [f'{Action.PICKUP.value}{i}' for i in range(num_passengers)] +
                          [Action.DROPOFF.value])


def __dropoff_specify(num_passengers):
    return __actions_dict(MOVE_STRINGS +
                          [Action.PICKUP.value] +
                          [f'{Action.DROPOFF.value}{i}' for i in range(num_passengers)])


def __pickup_and_dropoff_specify(num_passengers):
    return __actions_dict(MOVE_STRINGS +
                          [f'{Action.PICKUP.value}{i}' for i in range(num_passengers)] +
                          [f'{Action.DROPOFF.value}{i}' for i in range(num_passengers)])


def __specify_with_refuel(specify_func, num_passengers):
    a = specify_func(num_passengers)
    a[len(a)] = Action.REFUEL.value
    return a


def __specify_with_refuel_and_standby(specify_func, num_passengers):
    a = specify_func(num_passengers)
    a[len(a)] = Action.STANDBY.value
    a[len(a)] = Action.REFUEL.value
    return a


STANDARD_GENERIC = __actions_dict(MOVE_STRINGS + [Action.PICKUP.value, Action.DROPOFF.value])
PICKUP_ONLY_GENERIC = __actions_dict(MOVE_STRINGS + [Action.PICKUP.value])
STANDARD_REFUEL = __actions_dict(MOVE_STRINGS + [Action.PICKUP.value, Action.DROPOFF.value, Action.REFUEL.value])
ENGINE_CTL = __actions_dict(MOVE_STRINGS + [Action.PICKUP.value, Action.DROPOFF.value] + ENGINE_STRS)

TRUE_POSSIBLE_ACTIONS = {
    'single_taxi_simple': {'taxi_0': STANDARD_GENERIC},
    'single_taxi_two_passenger': {'taxi_0': STANDARD_GENERIC},
    'single_taxi_three_passenger': {'taxi_0': STANDARD_GENERIC},
    'single_taxi_four_passenger': {'taxi_0': STANDARD_GENERIC},
    'single_taxi_five_passenger': {'taxi_0': STANDARD_GENERIC},
    'single_taxi_two_passenger_pickup_only': {'taxi_0': PICKUP_ONLY_GENERIC},
    'single_taxi_three_passenger_pickup_only': {'taxi_0': PICKUP_ONLY_GENERIC},
    'single_taxi_four_passenger_pickup_only': {'taxi_0': PICKUP_ONLY_GENERIC},
    'single_taxi_five_passenger_pickup_only': {'taxi_0': PICKUP_ONLY_GENERIC},
    'single_taxi_fuel': {'taxi_0': STANDARD_REFUEL},
    'single_taxi_engine_control': {'taxi_0': ENGINE_CTL},
    'single_taxi_specify_pickup': {'taxi_0': __pickup_specify(3)},
    'single_taxi_specify_dropoff': {'taxi_0': __dropoff_specify(3)},
    'single_taxi_specify_pickup_and_dropoff': {'taxi_0': __pickup_and_dropoff_specify(3)},
    'single_taxi_small_fuel_map': {'taxi_0': STANDARD_GENERIC},
    'single_taxi_simple_small_map_no_obs': {'taxi_0': STANDARD_GENERIC},
    'two_taxi_simple': {f'taxi_{i}': STANDARD_GENERIC for i in range(2)},
    'three_taxi_simple': {f'taxi_{i}': STANDARD_GENERIC for i in range(3)},
    'four_taxi_simple': {f'taxi_{i}': STANDARD_GENERIC for i in range(4)},
    'five_taxi_simple': {f'taxi_{i}': STANDARD_GENERIC for i in range(5)},
    'two_taxi_see_each_other': {f'taxi_{i}': STANDARD_GENERIC for i in range(2)},
    'three_taxi_see_each_other': {f'taxi_{i}': STANDARD_GENERIC for i in range(3)},
    'four_taxi_see_each_other': {f'taxi_{i}': STANDARD_GENERIC for i in range(4)},
    'five_taxi_see_each_other': {f'taxi_{i}': STANDARD_GENERIC for i in range(5)},
    'two_taxi_fuel_limit': {
        'taxi_0': STANDARD_REFUEL,
        'taxi_1': STANDARD_REFUEL,
    },
    'three_taxi_fuel_limit': {
        'taxi_0': STANDARD_REFUEL,
        'taxi_1': STANDARD_REFUEL,
        'taxi_2': STANDARD_REFUEL
    },
    'two_taxi_small_map': {f'taxi_{i}': STANDARD_GENERIC for i in range(2)},
    'two_taxi_see_each_other_small_map': {f'taxi_{i}': STANDARD_GENERIC for i in range(2)},
    'single_taxi_image_observation': {'taxi_0': __pickup_specify(2)},
    'single_taxi_multi_observation': {'taxi_0': __pickup_specify(2)},
    'three_taxi_multi_observation': {
        'taxi_0': __specify_with_refuel_and_standby(__pickup_specify, 2),
        'taxi_1': __specify_with_refuel(__pickup_specify, 2),
        'taxi_2': __pickup_specify(2)
    }
}


@pytest.mark.parametrize(['env_name', 'env_cfg'], test_env_cfgs.items())
def test_action_spaces(env_name, env_cfg):
    assert env_name in TRUE_POSSIBLE_ACTIONS, f'missing ground-truth actions for env: {env_name}'

    check_action_space_validity(env_cfg, TRUE_POSSIBLE_ACTIONS[env_name])


def check_action_space_validity(taxi_env_params, expected_possible_actions):
    env = multi_taxi_v0.parallel_env(**taxi_env_params)

    # check correct action space and meanings
    for agent in env.possible_agents:
        action_space = env.action_space(agent)
        action_meanings = env.unwrapped.get_action_meanings(agent)
        expected_possible_actions_for_agent = expected_possible_actions[agent]

        assert action_meanings == expected_possible_actions_for_agent
        assert action_space == Discrete(n=len(expected_possible_actions_for_agent))
