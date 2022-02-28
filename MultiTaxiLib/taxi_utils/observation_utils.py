"""
This file will contain observations functions for the environment to use.
The use is of the following:
The environment is passed a function via the control_config.py file, and uses this function to calculate the observation of each
agent at each turn.
The different observations functions are implemented in this file.
"""
import numpy as np
from MultiTaxiLib.taxi_utils import rendering_utils


def partial_observations(state: list) -> list:
    """
    Given a state, return a list of vector observations, each observation is the vector observation defined by:
    [taxi_row, taxi_col, taxi_fuel_status, passengers_start_locations, passengers_destinations, passengers_status].
    Args:
        state: state of the domain (taxis, fuels, passengers_start_coordinates, destinations, passengers_locations)

    Returns: list of observations s.t each taxi sees only itself

    """

    def flatten(x):
        return [item for sub in x for item in sub]

    observations = []
    taxis, fuels, passengers_start_locations, passengers_destinations, passengers_locations = state
    pass_info = flatten(passengers_start_locations) + flatten(passengers_destinations) + passengers_locations

    for i in range(len(taxis)):
        obs = taxis[i] + [fuels[i]] + pass_info
        obs = np.reshape(obs, [1, len(obs)])
        observations.append(obs)

    return observations


def get_status_vector_observation(state: list, agent_name: str, taxis_names: list, num_taxis: int,
                                  can_see_others: bool = False) -> np.array:
    """
    Takes only the observation of the specified agent.
    Args:
        can_see_others: if True - each taxi will get other taxis locations
        state: state of the domain (taxis, fuels, passengers_start_coordinates, destinations, passengers_locations)
        agent_name: observer name
        taxis_names: list of all taxis names sorted
        num_taxis: number of taxis occupying the environment

    Returns: observation of the specified agent (state wise)

    """

    def flatten(x):
        return [item for sub in list(x) for item in list(sub)]

    agent_index = taxis_names.index(agent_name)

    taxis, fuels, passengers_start_locations, passengers_destinations, passengers_locations = state.copy()
    passengers_information = flatten(passengers_start_locations) + flatten(
        passengers_destinations) + passengers_locations

    observations = taxis[agent_index].copy()
    if can_see_others:
        for index in range(num_taxis):
            if index != agent_index:
                observations += taxis[index].copy()

    # observations += [0, 0] * (num_taxis - 1) + [fuels[agent_index]] + [0] * (num_taxis - 1) + passengers_information
    observations += [fuels[agent_index]] + passengers_information
    observations = np.reshape(observations, -1)

    return observations


def get_status_vector_observation_with_passenger_allocation(state: list, agent_name: str, taxis_names: list,
                                                            num_taxis: int) -> np.array:
    """
    Takes only the observation of the specified agent.
    Args:
        state: state of the domain (taxis, fuels, passengers_start_coordinates, destinations, passengers_locations)
        agent_name: observer name
        taxis_names: list of all taxis names sorted
        num_taxis: number of taxis occupying the environment

    Returns: observation of the specified agent (state wise)

    """

    def flatten(x):
        return [item for sub in list(x) for item in list(sub)]

    agent_index = taxis_names.index(agent_name)

    taxis, fuels, passengers_start_locations, passengers_destinations, passengers_locations = state.copy()
    passengers_information = passengers_start_locations[agent_index] + passengers_destinations[agent_index] + \
                             [passengers_locations[agent_index]]

    observations = taxis[agent_index].copy()

    # observations += [0, 0] * (num_taxis - 1) + [fuels[agent_index]] + [0] * (num_taxis - 1) + passengers_information
    observations += [fuels[agent_index]] + passengers_information
    observations = np.reshape(observations, -1)

    return observations


def get_image_obs_by_agent_id(agent_id: int, state: list, num_taxis: int, collided: np.ndarray, view_len: int,
                              domain_map: list) -> np.ndarray:
    """
    Returns an RGB image of the observation window of the current agent"
    Args:
        agent_id: taxi id
        state: current state of the environment
        num_taxis: number of taxis occupying the environment
        collided: list of indicators of collision for each taxi
        view_len: height and width diameter of the observation window
        domain_map: grid world representation of the world

    Returns: RGB image of shape (len_view, len_view, 3)

    """
    taxis_locations, _, passengers_start_locations, destinations, passengers_status = state
    rgb_map = rendering_utils.map2rgb(state, rendering_utils.get_current_map_with_agents(
        domain_map, state, num_taxis, collided))

    location = taxis_locations[agent_id]
    row = location[0] + 1
    col = location[1] * 2 + 1

    padded_map = np.zeros((rgb_map.shape[0] + view_len * 2, rgb_map.shape[1] + view_len * 4, rgb_map.shape[2]))
    padded_map[view_len: view_len + rgb_map.shape[0], view_len * 2: view_len * 2 + rgb_map.shape[1]] = rgb_map

    # return rgb_map[row - view_len: row + view_len + 1, col - view_len: col + view_len + 1]
    padded_map = np.array(padded_map).astype(int)
    return padded_map[row: row + 2 * view_len + 1, col: col + 4 * view_len + 1, :]
