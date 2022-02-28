"""
This file contains the actual performing of actions.
It was created in order to clean the environment clean and short as possible.
"""
from typing import Optional, List

import numpy as np
from MultiTaxiLib.taxi_utils import basic_utils
from MultiTaxiLib.taxi_utils.basic_utils import is_there_a_path


def engine_is_off_actions(state: list, action: str, taxi: int, reward_method: callable, engine_status_list: list) -> (int, list):
    """
    Returns the reward according to the requested action given that the engine's is currently off.
    Also turns engine on if requested.
    Args:
        state: current state of the environment
        action: requested action
        taxi: index of the taxi specified, relevant for turning engine on
        reward_method: the reward function the domain uses
        engine_status_list: engine status list to use in case the action is "turn_on"
    Returns: correct reward

    """
    reward = reward_method(state, 'unrelated_action')
    if action == 'standby':  # standby while engine is off
        reward = reward_method('standby_engine_off')
    elif action == 'turn_engine_on':  # turn engine on
        reward = reward_method(state, 'turn_engine_on')
        engine_status_list[taxi] = 1

    return reward, engine_status_list


def take_movement(action: str, row: int, col: int, num_rows: int, num_columns: int, domain_grid_map: np.ndarray,
                  dest_location: Optional[List] = None, column_partitions: Optional[List] = None) -> (bool, int, int):
    """
    Takes a movement with regard to a specific location of a taxi,
    Args:
        action: direction to move
        row: current row
        col: current col
        num_rows:total number of rows and columns in the grid map
        num_columns: as num_rows
        domain_grid_map: domain map in grid representation
        dest_location: destination requested for "goto" actions
        column_partitions: list of columns partitions

    Returns: if moved (false if there is a wall), new row, new col

    """
    moved = False
    new_row, new_col = row, col
    max_row = num_rows - 1
    max_col = num_columns - 1

    if action == 'south':  # south
        if row != max_row:
            moved = True
        new_row = min(row + 1, max_row)
    elif action == 'north':  # north
        if row != 0:
            moved = True
        new_row = max(row - 1, 0)
    if action == 'east' and domain_grid_map[1 + row, 2 * col + 2] == b":":  # east
        if col != max_col:
            moved = True
        new_col = min(col + 1, max_col)
    elif action == 'west' and domain_grid_map[1 + row, 2 * col] == b":":  # west
        if col != 0:
            moved = True
        new_col = max(col - 1, 0)
    elif 'goto' in action:
        if is_there_a_path(column_partitions, [row, col], dest_location):
            moved = True
            new_row, new_col = dest_location

    return moved, new_row, new_col


def check_action_for_collision(state: list, taxi_index: int, taxis_locations: list, current_row: int, current_col: int,
                                moved: bool, current_action: int, current_reward: int, num_taxis: int,
                               option_to_standby: bool, action_index_dictionary: dict,
                               collided: np.ndarray, reward_method: callable) -> (int, bool, int, list, list):
    """
    Takes a desired location for a taxi and update it with regard to collision check.
    Args:
        state: current state of the environment
        taxi_index: index of the taxi
        taxis_locations: locations of all other taxis.
        current_row: of the taxi
        current_col: of the taxi
        moved: indicator variable
        current_action: the current action requested
        current_reward: the current reward (left unchanged if there is no collision)
        num_taxis: number of taxis occupying the environment
        option_to_standby: is the environment supports the option to standby, if True - if a collision is to happened,
        the later taxi to take action is changed to standby
        action_index_dictionary: {action_name: index_in_action_list}
        collided: a list indicating collission status of the taxis
        reward_method: a reward method to calculate reward with

    Returns: new_reward, new_moved, new_action_index

    """
    reward = current_reward
    row, col = current_row, current_col
    moved = moved
    action = current_action
    taxi = taxi_index
    # Check if the number of taxis on the destination location is greater than 0
    if len([i for i in range(num_taxis) if taxis_locations[i] == [row, col]]) > 0:
        if option_to_standby:
            moved = False
            action = action_index_dictionary['standby']
        else:
            collided[[i for i in range(len(taxis_locations)) if taxis_locations[i] == [row, col]]] = 1
            collided[taxi] = 1
            reward = reward_method(state, 'collision')
            taxis_locations[taxi] = [row, col]

    return reward, moved, action, taxis_locations, collided


def make_pickup(state: list, taxi: int, passengers_start_locations: list, passengers_status: list,
                 taxi_location: list, reward: int, taxis_capacity: list, reward_method: callable) -> (list, int):
    """
    Make a pickup (successful or fail) for a given taxi.
    Args:
        state: current state of the environment
        taxi: index of the taxi
        passengers_start_locations: current locations of the passengers
        passengers_status: list of passengers statuses (1, 2, greater..)
        taxi_location: location of the taxi
        reward: current reward
        taxis_capacity: a list of taxis maximum capacity
        reward_method: a reward method to calculate reward with

    Returns: updates passengers status list, updates reward

    """
    passengers_status = passengers_status
    reward = reward
    successful_pickup = False
    for i, location in enumerate(passengers_status):
        # Check if we can take this passenger
        if location == 2 and taxi_location == passengers_start_locations[i] and \
                basic_utils.is_there_place_on_taxi(passengers_status, taxi, taxis_capacity):
            passengers_status[i] = taxi + 3
            successful_pickup = True
            reward = reward_method(state, 'pickup')
    if not successful_pickup:  # passenger not at location
        reward = reward_method(state, 'bad_pickup')

    return passengers_status, reward


def make_dropoff(state: list, taxi: int, current_passengers_start_locations: list, current_passengers_status: list,
                  destinations: list, taxi_location: list, reward: int, passenger_index: int,
                 reward_method: callable) -> (list, list, int):
    """
    Make a dropoff (successful or fail) for a given taxi.
    Args:
        state: current state of the environment
        taxi: index of the taxi
        current_passengers_start_locations: current locations of the passengers
        current_passengers_status: list of passengers statuses (1, 2, greater..)
        destinations: list of passengers destinations
        taxi_location: location of the taxi
        reward: current reward
        passenger_index: index of the desired dropped off passenger
        reward_method: the reward method to use when returning the reward

    Returns: updates passengers status list, updated passengers start location, updates reward

    """
    reward = reward
    passengers_start_locations = current_passengers_start_locations.copy()
    passengers_status = current_passengers_status.copy()
    successful_dropoff = False
    # Check if we have the passenger and we are at his destination
    if passengers_status[passenger_index] == (taxi + 3) and taxi_location == destinations[passenger_index]:
        passengers_status[passenger_index] = 1
        reward = reward_method(state, 'final_dropoff', taxi)
        passengers_start_locations[passenger_index] = taxi_location
        successful_dropoff = True
    elif passengers_status[passenger_index] == (taxi + 3):  # drops off passenger not at destination
        passengers_status[passenger_index] = 2
        successful_dropoff = True
        reward = reward_method(state, 'intermediate_dropoff', taxi)
        passengers_start_locations[passenger_index] = taxi_location
    if not successful_dropoff:  # not carrying a passenger
        reward = reward_method(state, 'bad_dropoff')

    return passengers_status, passengers_start_locations, reward

# def _make_dropoff_(self, taxi: int, current_passengers_start_locations: list, current_passengers_status: list,
#                    destinations: list, taxi_location: list, reward: int) -> (list, list, int):
#     """
#     Make a dropoff (successful or fail) for a given taxi.
#     Args:
#         taxi: index of the taxi
#         current_passengers_start_locations: current locations of the passengers
#         current_passengers_status: list of passengers statuses (1, 2, greater..)
#         destinations: list of passengers destinations
#         taxi_location: location of the taxi
#         reward: current reward
#
#     Returns: updates passengers status list, updated passengers start location, updates reward
#
#     """
#     reward = reward
#     passengers_start_locations = current_passengers_start_locations.copy()
#     passengers_status = current_passengers_status.copy()
#     successful_dropoff = False
#     for i, location in enumerate(passengers_status):  # at destination
#         location = passengers_status[i]
#         # Check if we have the passenger and we are at his destination
#         if location == (taxi + 3) and taxi_location == destinations[i]:
#             passengers_status[i] = 1
#             reward = self.partial_closest_path_reward('final_dropoff', taxi)
#             passengers_start_locations[i] = taxi_location
#             successful_dropoff = True
#             break
#         elif location == (taxi + 3):  # drops off passenger not at destination
#             passengers_status[i] = 2
#             successful_dropoff = True
#             reward = self.partial_closest_path_reward('intermediate_dropoff', taxi)
#             passengers_start_locations[i] = taxi_location
#             break
#     if not successful_dropoff:  # not carrying a passenger
#         reward = self.partial_closest_path_reward('bad_dropoff')
#
#     return passengers_status, passengers_start_locations, reward


def update_movement_wrt_fuel(state: list, taxi: int, taxis_locations: list, wanted_row: int, wanted_col: int, reward: int,
                             fuel: int, is_infinite_fuel: bool, reward_method: callable) -> (int, int, list):
    """
    Given that a taxi would like to move - check the fuel accordingly and update reward and location.
    Args:
        state: current state of the environment
        taxi: index of the taxi
        taxis_locations: list of current locations (prior to movement)
        wanted_row: row after movement
        wanted_col: col after movement
        reward: current reward
        fuel: current fuel
        is_infinite_fuel: an indicator variable whether fuel is limited or not
        reward_method: a reward method to calculate the reward with

    Returns: updated_reward, updated fuel, updated_taxis_locations

    """
    reward = reward
    fuel = fuel
    taxis_locations = taxis_locations
    if fuel == 0 and (not is_infinite_fuel):
        reward = reward_method(state, 'bad_refuel')
    else:
        fuel = max(0, fuel - 1)
        taxis_locations[taxi] = [wanted_row, wanted_col]

    return reward, fuel, taxis_locations


def refuel_taxi(state: list, current_fuel: int, current_reward: int, taxi: int, taxis_locations: list, fuel_stations: list,
                 domain_grid_map: list, fuel_type_list: list, is_infinite_fuel: bool, max_fuel: list,
                 reward_method: callable) -> (int, int):
    """
    Try to refuel a taxi, if successful - updates fuel tank, if not - updates the reward.
    Args:
        state: current state f the environment
        current_fuel: current fuel of the taxi
        current_reward: current reward for the taxi.
        taxi: taxi index
        taxis_locations: list of current taxis locations
        fuel_stations: list of fuel stations coordinations
        domain_grid_map: list with a grid map representation of the domain
        fuel_type_list: list of taxis fuel types
        is_infinite_fuel: indicator variable whether fuel is limited or not
        max_fuel: list with max fuel limit of each taxi
        reward_method: a reward method to calculate reward with


    Returns: updated reward, updated fuel

    """
    fuel = current_fuel
    reward = current_reward
    if basic_utils.at_valid_fuel_station(taxi, taxis_locations, fuel_stations, domain_grid_map,
                                         fuel_type_list) and not is_infinite_fuel:
        fuel = max_fuel[taxi]
    else:
        reward = reward_method(state, 'bad_refuel')

    return reward, fuel
