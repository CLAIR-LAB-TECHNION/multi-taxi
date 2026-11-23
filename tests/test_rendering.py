import pytest

from multi_taxi import multi_taxi_v0


@pytest.mark.parametrize('collided, clear_dead_taxis, at_destination, passenger_shown', [
    (False, True, False, False),  # live taxi hides its passenger
    (True, False, False, False),  # dead taxi still drawn hides its passenger
    (True, True, False, True),  # cleared dead taxi leaves its passenger visible
    (True, True, True, True),  # passenger is drawn over its own destination
])
def test_render_passenger_in_taxi(collided, clear_dead_taxis, at_destination, passenger_shown):
    env = multi_taxi_v0.parallel_env(num_taxis=2, num_passengers=1, can_collide=True,
                                     clear_dead_taxis=clear_dead_taxis, passenger_fumble=False, render_mode='ansi')
    env.reset(seed=0)
    s = env.state()
    loc = s.passengers[0].destination if at_destination else (0, 0)
    s.taxis[0].location = loc
    s.taxis[0].collided = collided
    s.taxis[1].location = (6, 11)  # a live taxi elsewhere
    s.passengers[0].location = loc
    s.taxis[0].pick_up(s.passengers[0])
    env.unwrapped.set_state(s)

    rendered_map = env.render().rsplit('+', 1)[0]  # drop the status text under the map
    assert ('P' in rendered_map) == passenger_shown
