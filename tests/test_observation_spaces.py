from gym.spaces import MultiDiscrete

from multi_taxi.taxi_environment import TaxiEnv, orig_MAP, simple_MAP, MAP2


def check_observation_space_validity(taxi_env_params, expected_observation_space):
    env = TaxiEnv(**taxi_env_params)
    obs = env.reset()
    # check correct observation space
    assert env.observation_space == MultiDiscrete(expected_observation_space)
    for single_taxi_obs in obs.values():  # check correct observation for each taxi
        assert len(single_taxi_obs) == len(env.observation_space)
        for obs_space_value, taxi_obs_value in zip(env.observation_space, single_taxi_obs):
            assert 0 <= taxi_obs_value < obs_space_value.n


def test_get_observation_space_list():
    # growing number of taxis
    check_observation_space_validity({}, [7, 12, 7, 12, 7, 12, 4])
    check_observation_space_validity(dict(num_taxis=2), [7, 12, 7, 12, 7, 12, 5])
    check_observation_space_validity(dict(num_taxis=3), [7, 12, 7, 12, 7, 12, 6])
    check_observation_space_validity(dict(num_taxis=4), [7, 12, 7, 12, 7, 12, 7])
    check_observation_space_validity(dict(num_taxis=5), [7, 12, 7, 12, 7, 12, 8])

    # taxis can see each other
    check_observation_space_validity(dict(num_taxis=2,
                                          can_see_others=True), [7, 12, 7, 12, 7, 12, 7, 12, 5])
    check_observation_space_validity(dict(num_taxis=3,
                                          can_see_others=True), [7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 6])
    check_observation_space_validity(dict(num_taxis=4,
                                          can_see_others=True), [7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 7])
    check_observation_space_validity(dict(num_taxis=5,
                                          can_see_others=True), [7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 8])

    # growing number of passengers
    check_observation_space_validity(dict(num_passengers=2), [7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 4, 4])
    check_observation_space_validity(dict(num_passengers=3), [7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 4, 4, 4])
    check_observation_space_validity(dict(num_passengers=4),
                                     [7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 4, 4, 4, 4])
    check_observation_space_validity(dict(num_passengers=5),
                                     [7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 4, 4,
                                      4, 4, 4])

    # pickup only
    check_observation_space_validity(dict(num_passengers=2,
                                          pickup_only=True), [7, 12, 7, 12, 7, 12, 4, 4])
    check_observation_space_validity(dict(num_passengers=3,
                                          pickup_only=True), [7, 12, 7, 12, 7, 12, 7, 12, 4, 4, 4])
    check_observation_space_validity(dict(num_passengers=4,
                                          pickup_only=True),
                                     [7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 4, 4, 4, 4])
    check_observation_space_validity(dict(num_passengers=5,
                                          pickup_only=True),
                                     [7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 4, 4, 4, 4, 4])

    # using fuel
    check_observation_space_validity(dict(num_taxis=2,
                                          max_fuel=[10, 5]), [7, 12, 11, 7, 12, 7, 12, 5])
    check_observation_space_validity(dict(num_taxis=3,
                                          max_fuel=[15, 25, 99],
                                          can_see_others=True), [7, 12, 7, 12, 7, 12, 100, 7, 12, 7, 12, 6])

    # using a different maps
    check_observation_space_validity(dict(num_taxis=2,
                                          domain_map=orig_MAP), [5, 5, 5, 5, 5, 5, 5])
    check_observation_space_validity(dict(num_taxis=2,
                                          domain_map=orig_MAP,
                                          can_see_others=True), [5, 5, 5, 5, 5, 5, 5, 5, 5])
    check_observation_space_validity(dict(domain_map=MAP2), [4, 4, 4, 4, 4, 4, 4])
    check_observation_space_validity(dict(domain_map=simple_MAP), [5, 5, 5, 5, 5, 5, 4])
