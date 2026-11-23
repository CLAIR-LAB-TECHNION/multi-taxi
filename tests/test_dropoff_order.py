import pytest

from multi_taxi import single_taxi_v0


@pytest.mark.parametrize('specify, dropoff_order, p0_arrived, p2_at_destination, action, arrived, riders', [
    (True, [0, 1], True, True, 'dropoff1', {0, 1}, {2}),  # order satisfied
    (True, [0, 1], False, True, 'dropoff1', set(), {1, 2}),  # passenger 0 not delivered yet
    (True, [0, 1, 2], False, False, 'dropoff2', set(), {1}),  # intermediate dropoff ignores the order
    (False, [0, 1], True, True, 'dropoff', {0, 1}, {2}),  # order checked against passenger 0, not the taxi's first rider
    (False, [0, 1, 2], False, True, 'dropoff', set(), {1, 2}),  # all riders blocked
    (False, [0, 1], False, False, 'dropoff', set(), {1}),  # blocked rider stays, other rider dropped elsewhere
])
def test_dropoff_order(specify, dropoff_order, p0_arrived, p2_at_destination, action, arrived, riders):
    env = single_taxi_v0.gym_env(num_passengers=3, dropoff_order=dropoff_order, specify_passenger_dropoff=specify)
    env.reset()
    action_meanings = env.unwrapped.get_action_meanings()

    # taxi at (0, 0) carries passengers 1 and 2. passenger 1's destination is (0, 0)
    s = env.state()
    taxi = s.taxis[0]
    taxi.location = (0, 0)
    p0, p1, p2 = s.passengers
    p0.location = p0.destination = (1, 1)
    if not p0_arrived:
        p0.location = (2, 2)
    p1.destination = (0, 0)
    p2.destination = (0, 0) if p2_at_destination else (3, 3)
    for p in (p1, p2):
        p.location = taxi.location
        taxi.pick_up(p)
    env.unwrapped.set_state(s)

    info = env.step([a for a in action_meanings if action_meanings[a] == action][0])[-1]
    s = env.state()
    assert info['dropoff_success'] == (riders != {1, 2})
    assert {p.id for p in s.passengers if p.arrived} == arrived
    assert {p.id for p in s.taxis[0].passengers} == riders
