"""
This util file contains basic utils regarding to the domain, such as: metrics, taxis calculations, map parsing...
"""

import numpy as np


def is_there_place_on_taxi(passengers_locations: np.array, taxi_index: int, taxis_capacity: list) -> bool:
    """
    Checks if there is room for another passenger on taxi number 'taxi_index'.
    Args:
        passengers_locations: list of all passengers locations
        taxi_index: index of the desired taxi
        taxis_capacity: list of ints indicating each taxi's capacity

    Returns: Whether there is a place (True) or not (False)

    """
    # Remember that passengers "location" is: 1 - delivered, 2 - waits for a taxi, >2 - on a taxi with index
    # equals to location-3 (where taxi0 is the first taxi and passengers with location "3" are on it)

    return (len([location for location in passengers_locations if location == (taxi_index + 3)]) <
            taxis_capacity[taxi_index])


def map_at_location(location: list, domain_map: list) -> str:
    """
    Returns the map character on the specified coordinates of the grid.
    Args:
        location: location to check [row, col]
        domain_map: grid world map representation

    Returns: character on specific location on the map

    """
    row, col = location[0], location[1]

    return domain_map[row + 1][2 * col + 1].decode(encoding='UTF-8')


def at_valid_fuel_station(taxi: int, taxis_locations: list, fuel_stations: list, domain_map: list,
                          fuel_type_list: list) -> bool:
    """
    Checks if the taxi's location is a suitable fuel station or not.
    Args:
        taxi: the index of the desired taxi
        taxis_locations: list of taxis coordinates [row, col]
        fuel_stations: list of fuel stations coordinations
        domain_map: grid map list representation
        fuel_type_list: list of fuel type of each taxi
    Returns: whether the taxi is at a suitable fuel station (true) or not (false)

    """
    return (taxis_locations[taxi] in fuel_stations and
            map_at_location(taxis_locations[taxi], domain_map) == fuel_type_list[
                taxi])


def get_action_list(action_list) -> list:
    """
    Return a list in the correct format for the step function that should
    always get a list even if it's a single action.
    Args:
        action_list:

    Returns: list(action_list)

    """
    if type(action_list) == int:
        return [action_list]
    elif type(action_list) == np.int64 or type(action_list) == np.int32:
        return [action_list]

    return action_list


def get_l1_distance(location1, location2):
    """
    Return the minimal travel length between 2 locations on the grid world.
    Args:
        location1: [i1, j1]
        location2: [i2, j2]

    Returns: np.abs(i1 - i2) + np.abs(j1 - j2)

    """
    return np.abs(location1[0] - location2[0]) + np.abs(location1[1] - location2[1])


def passenger_destination_l1_distance(state: list, passenger_index, current_row: int, current_col: int) -> int:
    """
    Returns the manhattan distance between passenger current defined "start location" and it's destination.
    Args:
        state: current state of the environment
        passenger_index: index of the passenger.
        current_row: current row to calculate distance from destination
        current_col: current col to calculate distance from destination

    Returns: manhattan distance

    """
    current_state = state
    destination_row, destination_col = current_state[3][passenger_index]

    return get_l1_distance([destination_row, destination_col], [current_row, current_col])


def get_array_as_list_type(array: np.ndarray) -> list:
    """
    Gets an array with more than 1-sim and return it as a list of lists.
    Args:
        array: array to modify

    Returns: list of lists constructed from the array

    """
    if isinstance(array, list):
        return array

    if len(array.shape) == 1:
        return list(array)

    return [list(element) for element in array]


def is_there_a_path(column_partitions: list, current_location: list, destination: list) -> bool:
    """
    Check if there is a possible path between the current location to the destination
    (basically check if there is a wall crossing between the 2 points).
    Args:
        column_partitions: list of column walls from top down
        current_location: source location
        destination: destination

    Returns: bool - True or False depends on the existence of a path

    """
    for partition in column_partitions:
        if (current_location[1] < partition < destination[1]) or (current_location[1] > partition > destination[1]):
            return False

    return True
