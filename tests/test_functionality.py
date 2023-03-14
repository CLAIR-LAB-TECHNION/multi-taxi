from multi_taxi import multi_taxi_v0, single_taxi_v0, maps, Action, FuelType

movements = {
    0: (1, 0),  # south
    1: (-1, 0),  # north
    2: (0, 1),  # east
    3: (0, -1)  # west
}


def test_navigation():
    env = single_taxi_v0.gym_env(domain_map=maps.HOURGLASS)  # use a map with blockades
    env.reset()

    # use domain map object to check transition validity
    domain_map = env.unwrapped.domain_map

    # iterate free cells
    for loc in domain_map.iter_free_locations():

        # put single taxi in free cell
        s = env.state()
        s.taxis[0].location = loc
        env.unwrapped.set_state(s)

        # try all possible navigation transitions and check if transitioned correctly
        map_row, map_col = domain_map.location_to_domain_map_idx(*loc)
        for action, (row_move, col_move) in movements.items():
            env.step(action)
            taxi = env.state().taxis[0]
            next_map_row = map_row + row_move
            next_map_col = map_col + col_move

            if taxi.location != loc:  # taxi moved

                if col_move != 0:  # horizontal.
                    assert (domain_map[next_map_row, next_map_col] != '|' and
                            domain_map[next_map_row, next_map_col + col_move] != 'X'), f'{loc}, {action}'
                else:  # vertical
                    assert (domain_map[next_map_row, next_map_col] != 'X' and
                            domain_map[next_map_row, next_map_col] != '-'), f'{loc}, {action}'
            else:
                # horizontal. check for walls and blockades
                if col_move != 0:
                    assert (domain_map[next_map_row, next_map_col] == '|' or
                            domain_map[next_map_row, next_map_col + col_move] == 'X'), f'{loc}, {action}'
                else:
                    assert (domain_map[next_map_row, next_map_col] == 'X' or
                            domain_map[next_map_row, next_map_col] == '-'), f'{loc}, {action}'

            # reset state
            env.unwrapped.set_state(s)


def test_refuel():
    # use a map with both kinds of fuel stations
    env = single_taxi_v0.gym_env(domain_map=maps.DEFAULT_MAP, max_fuel=10, fuel_type=FuelType.FUEL)
    env.reset()
    action_meanings = env.unwrapped.get_action_meanings()
    refuel_action = [i for i in action_meanings if action_meanings[i] == Action.REFUEL.value][0]

    s = env.state()
    s.taxis[0].location = (0, 2)
    s.taxis[0].fuel = 3
    env.unwrapped.set_state(s)

    env.step(refuel_action)
    taxi = env.state().taxis[0]
    assert taxi.fuel == 10


def test_bad_fuel():
    # use a map with both kinds of fuel stations
    env = single_taxi_v0.gym_env(domain_map=maps.DEFAULT_MAP, max_fuel=10, fuel_type=FuelType.GAS)
    env.reset()
    action_meanings = env.unwrapped.get_action_meanings()
    refuel_action = [i for i in action_meanings if action_meanings[i] == Action.REFUEL.value][0]

    s = env.state()
    s.taxis[0].location = (0, 2)
    s.taxis[0].fuel = 3
    env.unwrapped.set_state(s)

    env.step(refuel_action)
    taxi = env.state().taxis[0]
    assert taxi.fuel == 2


def test_swap_collision():
    env = multi_taxi_v0.env(num_taxis=2, can_collide=True)
    env.reset()
    s = env.state()
    s.taxis[0].location = (0, 0)
    s.taxis[1].location = (1, 0)
    env.unwrapped.set_state(s)

    env.step(0)
    env.step(1)

    taxis = env.state().taxis
    assert all(taxi.collided for taxi in taxis)


def test_same_cell_collision():
    env = multi_taxi_v0.env(num_taxis=2, can_collide=True)
    env.reset()
    s = env.state()
    s.taxis[0].location = (0, 0)
    s.taxis[1].location = (2, 0)
    env.unwrapped.set_state(s)

    env.step(0)
    env.step(1)

    taxis = env.state().taxis
    assert all(taxi.collided for taxi in taxis)


def test_triple_collision():
    env = multi_taxi_v0.env(num_taxis=3, can_collide=True)
    env.reset()
    s = env.state()
    s.taxis[0].location = (2, 2)
    s.taxis[1].location = (2, 3)
    s.taxis[2].location = (3, 3)
    env.unwrapped.set_state(s)

    env.step(2)
    env.step(0)
    env.step(1)

    taxis = env.state().taxis
    assert all(taxi.collided for taxi in taxis)


def test_non_collider():
    env = multi_taxi_v0.env(num_taxis=3, can_collide=[True, True, False])
    env.reset()
    s = env.state()
    s.taxis[0].location = (2, 2)
    s.taxis[1].location = (2, 3)
    s.taxis[2].location = (3, 3)
    env.unwrapped.set_state(s)

    env.step(2)
    env.step(0)
    env.step(1)

    taxis = env.state().taxis
    assert all(not taxi.collided for taxi in taxis)
