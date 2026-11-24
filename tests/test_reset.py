import pytest

from multi_taxi import multi_taxi_v0


@pytest.mark.parametrize('distinct_dropoffs', [True, False])
@pytest.mark.parametrize('num_passengers', [1, 2, 5])
def test_no_passenger_starts_at_destination(num_passengers, distinct_dropoffs):
    env = multi_taxi_v0.parallel_env(num_passengers=num_passengers,
                                     distinct_passenger_initial_pickups=False,
                                     distinct_passenger_dropoffs=distinct_dropoffs)
    for seed in range(200):
        env.reset(seed=seed)
        passengers = env.unwrapped.state().passengers
        assert not any(p.arrived for p in passengers), seed
        if distinct_dropoffs:
            assert len({p.destination for p in passengers}) == num_passengers, seed
