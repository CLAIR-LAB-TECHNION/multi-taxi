import pytest

from multi_taxi import multi_taxi_v0

# taxi_0 carries passenger 0 from (0, 0) east to (0, 1) and dies on arrival
DEATH_CONFIGS = {
    'crash': dict(can_collide=[True, True, False]),
    'fuel': dict(max_fuel=[1, None, None]),
    'steps': dict(max_steps=[1, None, None]),
}


def _act(env, **actions):
    return {agent: env.unwrapped.get_action_map(agent)[actions.get(agent, 'pickup')] for agent in env.agents}


@pytest.mark.parametrize('death', DEATH_CONFIGS)
@pytest.mark.parametrize('fumble', [None, False])
def test_passengers_of_dead_taxi(death, fumble):
    env = multi_taxi_v0.parallel_env(num_taxis=3, num_passengers=1, passenger_fumble=fumble, **DEATH_CONFIGS[death])
    env.reset()
    s = env.state()
    for taxi, loc in zip(s.taxis, [(0, 0), (0, 1), (0, 1)]):
        taxi.location = loc
    s.passengers[0].location = (0, 0)
    s.passengers[0].destination = (6, 0)
    s.taxis[0].pick_up(s.passengers[0])

    [(new_state, *_)] = env.unwrapped.state_action_transitions(s, _act(env, taxi_0='east'))
    env.unwrapped.set_state(new_state)
    env.step(_act(env))

    p = env.state().passengers[0]
    if fumble is None:  # default
        assert new_state == new_state.copy()
        assert p.location == (0, 1) and p.carrying_taxi in (1, 2)
    else:
        assert p.carrying_taxi == 0


def test_pickup_only_objective_kept_when_taxi_dies():
    env = multi_taxi_v0.parallel_env(num_taxis=2, num_passengers=2, pickup_only=True, max_fuel=[1, None])
    env.reset()
    s = env.state()
    for taxi, p, loc in zip(s.taxis, s.passengers, [(0, 0), (3, 3)]):
        taxi.location = p.location = loc
    s.taxis[0].pick_up(s.passengers[0])
    env.unwrapped.set_state(s)

    _, _, terms, _, _ = env.step(_act(env, taxi_0='east'))
    assert env.unwrapped.objective_achieved() and all(terms.values())
