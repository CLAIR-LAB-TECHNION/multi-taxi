from multi_taxi.taxi_utils import basic_utils


def get_done_dictionary(current_done_dictionary: dict, taxi: int, taxis_locations: list, fuel_stations: list,
                        fuel_type_list: list, desc, passengers_status: list, fuel_tanks: list, collision_status: dict,
                        is_infinite_fuel: bool, taxi_names: list, pickup_only: bool) -> dict:
    """
    Get passengers statuses, fuel indications, collided indications and retrieve the updated done dictionary
    Args:
        current_done_dictionary: the current done statuses of all taxis
        passengers_status: whether passengers arrived or not
        fuel_tanks: fuel status of all taxis
        collision_status: whether taxis collided or not
        is_infinite_fuel: are there fuel limitations
        taxi_names: agent names

    Returns: the updated done status for each taxi

    """
    updated_dones = current_done_dictionary

    if not is_infinite_fuel:
        for i, fuel in enumerate(fuel_tanks):
            if fuel <= 0 and not basic_utils.at_valid_fuel_station(taxi, taxis_locations, fuel_stations,
                                                                   desc.copy().tolist(),
                                                                   fuel_type_list):
                updated_dones[taxi_names[i]] = True

    for i, collided in enumerate(collision_status):
        if collided == 1:
            updated_dones[taxi_names[i]] = True

    updated_dones['__all__'] = all(list(updated_dones.values()))

    if updated_dones['__all__']:
        return updated_dones

    are_all_passengers_arrived = True
    for status in passengers_status:
        if pickup_only:  # pickup only task, passenger not yet picked up
            if status == 2:
                are_all_passengers_arrived = False
        elif status != 1:  # pickup and dropoff task, passenger not in destination
            are_all_passengers_arrived = False
    if are_all_passengers_arrived:
        for key in list(updated_dones.keys()):
            updated_dones[key] = True

    return updated_dones
