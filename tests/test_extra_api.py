from multi_taxi import single_taxi_v0, maps

def test_get_domain_map():
    env = single_taxi_v0.gym_env(domain_map=maps.HOURGLASS)  # use a map with blockades
    env.reset()

    map1 = env.get_domain_map()
    map1[0][0] = '^!^'

    map2 = env.get_domain_map()

    assert map2[0][0] != map1[0][0]
