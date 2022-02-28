"""
This file will contain rewards functions for the environment to use.
The use is of the following:
The environment is passed a function via the control_config.py file, and uses this function to calculate the rewards of each
turn.
The different reward functions are implemented in this file.
"""
from MultiTaxiLib.config import TAXI_ENVIRONMENT_REWARDS
from MultiTaxiLib.taxi_utils import basic_utils


def partial_closest_path_reward(state: list, basic_reward_str: str, taxi_index: int = None,
                                reward_const: int = 15) -> int:
    """
    Computes the reward for a taxi and it's defined by:
    dropoff[s] - gets the reward equal to the closest path multiply by 15, if the drive got a passenger further
    away - negative.
    other actions - basic reward from config table
    Args:
        state: current state of the environment

        basic_reward_str: the reward we would like to give
        taxi_index: index of the specific taxi, using this parameter only if the action is a dropoff variation.
        reward_const: a hyper-parameter for the reward calculation, maybe negligible, need to check.

    Returns: updated reward

    """

    if basic_reward_str not in ['intermediate_dropoff', 'final_dropoff'] or taxi_index is None:
        return TAXI_ENVIRONMENT_REWARDS[basic_reward_str]

    # [taxis_locations, fuels, passengers_start_locations, destinations, passengers_status]
    current_state = state

    taxis_locations, _, passengers_start_locations, _, passengers_status = current_state
    passenger_index = passengers_status.index(taxi_index + 3)
    passenger_start_row, passenger_start_col = passengers_start_locations[passenger_index]
    taxi_current_row, taxi_current_col = taxis_locations[taxi_index]

    return (basic_utils.passenger_destination_l1_distance(state, passenger_index,
                                                          passenger_start_row, passenger_start_col) -
            basic_utils.passenger_destination_l1_distance(state, passenger_index,
                                                          taxi_current_row,
                                                          taxi_current_col)) * \
           TAXI_ENVIRONMENT_REWARDS[basic_reward_str] -1
