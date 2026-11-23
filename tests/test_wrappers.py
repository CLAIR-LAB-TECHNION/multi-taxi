import numpy as np
import pytest

from multi_taxi import multi_taxi_v0
from multi_taxi.wrappers import (FixedPassengerStartLocationsWrapper, FixedPassengerDestinationsWrapper,
                                 FixedTaxiStartLocationsWrapper)


@pytest.mark.parametrize(['wrapper', 'get_loc'], [
    (FixedPassengerStartLocationsWrapper, lambda s: s.passengers[0].location),
    (FixedPassengerDestinationsWrapper, lambda s: s.passengers[0].destination),
    (FixedTaxiStartLocationsWrapper, lambda s: s.taxis[0].location),
])
def test_fixed_locations_reset(wrapper, get_loc):
    env = wrapper(multi_taxi_v0.parallel_env(num_taxis=1, num_passengers=1), 1, 2)
    obs, infos = env.reset(seed=0, options={})

    assert tuple(get_loc(env.state())) == (1, 2)
    assert infos == {agent: {} for agent in env.agents}
    for agent, agent_obs in env.unwrapped.observe_all().items():
        np.testing.assert_array_equal(obs[agent], agent_obs)
