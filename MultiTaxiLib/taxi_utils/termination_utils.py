
def get_done_dictionary(current_done_dictionary: dict, passengers_status: list, fuel_tanks: list,
                        collision_status: dict, is_infinite_fuel: bool, taxi_names: list) -> dict:
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
            if fuel <= 0:
                updated_dones[taxi_names[i]] = True

    for i, collided in enumerate(collision_status):
        if collided == 1:
            updated_dones[taxi_names[i]] = True

    updated_dones['__all__'] = True
    updated_dones['__all__'] = all(list(updated_dones.values()))

    if updated_dones['__all__']:
        return updated_dones

    are_all_passengers_arrived = True
    for status in passengers_status:
        if status != 1:
            are_all_passengers_arrived = False
    if are_all_passengers_arrived:
        for key in list(updated_dones.keys()):
            updated_dones[key] = True

    return updated_dones
