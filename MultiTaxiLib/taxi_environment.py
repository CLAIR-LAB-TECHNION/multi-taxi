# -*- coding: utf-8 -*-
import random
from typing import Dict

import gym
from gym.utils import seeding
import numpy as np
from ray import rllib

from MultiTaxiLib.config import TAXI_ENVIRONMENT_REWARDS, BASE_AVAILABLE_ACTIONS, ALL_ACTIONS_NAMES
from gym.spaces import MultiDiscrete, Box

from MultiTaxiLib.taxi_utils import actions_utils, observation_utils, basic_utils, reward_utils, rendering_utils
from MultiTaxiLib.taxi_utils.rendering_utils import render
from MultiTaxiLib.taxi_utils.termination_utils import get_done_dictionary

MAP2 = [
    "+-------+",
    "|X: |F:X|",
    "| : | : |",
    "| : : : |",
    "|X| :G|X|",
    "+-------+",
]

MAP = [
    "+-----------------------+",
    "|X: |F: | : | : | : |F:X|",
    "| : | : : : | : | : | : |",
    "| : : : : : : : : : : : |",
    "| : : : : : | : : : : : |",
    "| : : : : : | : : : : : |",
    "| : : : : : : : : : : : |",
    "|X| :G| | | :G| | | : |X|",
    "+-----------------------+",
]

simple_MAP = [
    "+---------+",
    "| : : : : |",
    "| : : : : |",
    "| : : : : |",
    "| : : : : |",
    "| : : : : |",
    "+---------+",
]

orig_MAP = [
    "+---------+",
    "|X: | : :X|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|X| : |X: |",
    "+---------+",
]


class TaxiEnv(rllib.env.MultiAgentEnv):
    """
    The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich, adjusted for MARL by Ofir Abu
    Description:
    There are designated sources and destinations of passengers (chosen randomly).
    When the episode starts, the taxis start off at a random square and the passenger is at a random location.
    The taxis drive to the passenger's location, picks up the passenger, drives to the passenger's destination
    (another one of the four specified locations), and then drops off the passenger. Once the passenger is dropped off,
    the episode ends.
    A taxi can pickup more than one passenger, capacity is an hyperparameter.

    Observations:
    A list (taxis, fuels, pass_start, destinations, pass_locs):
        taxis:                  a list of coordinates of each taxi
        fuels:                  a list of fuels for each taxi
        pass_start:             a list of starting coordinates for each passenger (current position or last available)
        destinations:           a list of destination coordinates for each passenger
        passengers_locations:   a list of locations of each passenger.
                                -1 means delivered
                                0 means not picked up
                                positive number means the passenger is in the corresponding taxi number

    Observations consists of the pixel map around the taxi and a status vector.

    Passenger start: coordinates of each of these
    - -1: In a taxi
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)

    Passenger location:
    - -1: delivered
    - 0: not in taxi
    - x: in taxi x (x is integer)

    Destinations: coordinates of each of these
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)

    Fuel:
     - 0 to np.inf: default with 10, init with 0 means infinity fuel amount

    Actions:
    Actions are given as a list, each element referring to one taxi's action. Each taxi has 7 actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: dropoff passenger
    - 6: turn engine on
    - 7: turn engine off
    - 8: standby
    - 9: refuel fuel tank


    Rewards:
    - Those are specified in the config file.

    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters (R, G, Y and B): locations for passengers and destinations
    Main class to be characterized with hyper-parameters.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, _=0, num_taxis: int = 1, num_passengers: int = 1, max_fuel: list = None,
                 domain_map: list = MAP, taxis_capacity: list = None, collision_sensitive_domain: bool = False,
                 fuel_type_list: list = None, option_to_stand_by: bool = False, view_len: int = 2,
                 rewards_table: Dict = TAXI_ENVIRONMENT_REWARDS, observation_type: str = 'symbolic',
                 can_see_others: bool = False):
        """
        Args:
            num_taxis: number of taxis in the domain
            num_passengers: number of passengers occupying the domain at initiailaization
            max_fuel: list of max (start) fuel, we use np.inf as default for fuel free taxi.
            domain_map: 2D - map of the domain
            taxis_capacity: max capacity of passengers in each taxi (list)
            collision_sensitive_domain: is the domain show and react (true) to collisions or not (false)
            fuel_type_list: list of fuel types of each taxi
            option_to_stand_by: can taxis simply stand in place
            view_len: width and height of the observation window for a taxi
            can_see_others: if True, taxis observations will include other taxis locations
        """
        # initializing rewards table
        self.can_see_others = can_see_others
        if rewards_table != TAXI_ENVIRONMENT_REWARDS:
            rewards_table = TAXI_ENVIRONMENT_REWARDS.update(rewards_table)
        self.rewards_table = rewards_table

        # Initializing default values
        self.current_step = 0
        self.is_infinite_fuel = False
        if max_fuel is None:
            self.max_fuel = [0] * num_taxis
            self.is_infinite_fuel = True
        else:
            self.max_fuel = max_fuel

        if domain_map is None:
            self.desc = np.asarray(MAP, dtype='c')
        else:
            self.desc = np.asarray(domain_map, dtype='c')

        self.column_partitions = []

        if taxis_capacity is None:
            self.taxis_capacity = [num_passengers] * num_taxis
        else:
            self.taxis_capacity = taxis_capacity

        if fuel_type_list is None:
            self.fuel_type_list = ['F'] * num_passengers
        else:
            self.fuel_type_list = fuel_type_list

        # Relevant features for map orientation, notice that we can only drive between the columns (':')
        self.num_rows = num_rows = len(self.desc) - 2
        self.num_columns = num_columns = len(self.desc[0][1:-1:2])

        # Set locations of passengers and fuel stations according to the map.
        self.passengers_locations = []
        self.fuel_station1 = None
        self.fuel_station2 = None
        self.fuel_stations = []

        self.fuel_stations = rendering_utils.parse_fuel_stations_from_map(self.desc)

        self.num_passengers = num_passengers

        self.coordinates = np.array([[i, j] for i in range(num_rows) for j in range(num_columns)])

        self.passengers_locations = list(self.coordinates[
                                         np.random.choice(self.coordinates.shape[0], size=2 * num_passengers,
                                                          replace=False), :])

        self.coordinates = basic_utils.get_array_as_list_type(self.coordinates)
        self.passengers_locations = basic_utils.get_array_as_list_type(self.passengers_locations)

        self.num_taxis = num_taxis
        self.taxis_names = ["taxi_" + str(index + 1) for index in range(num_taxis)]

        self.collision_sensitive_domain = collision_sensitive_domain

        # Indicator list of 1's (collided) and 0's (not-collided) of all taxis
        self.collided = np.zeros(num_taxis)

        self.option_to_standby = option_to_stand_by

        # A list to indicate whether the engine of taxi i is on (1) or off (0), all taxis start as on.
        self.engine_status_list = list(np.ones(num_taxis).astype(bool))

        # Available actions in relation to all actions based on environment parameters.
        self.available_actions_indexes, self.index_action_dictionary, self.action_index_dictionary \
            = self._set_available_actions_dictionary()
        self.num_actions = len(self.available_actions_indexes)
        # self.action_space = gym.spaces.Discrete(self.num_actions)
        self.action_space = gym.spaces.Discrete(self.num_actions)
        # self.obs_space = gym.spaces.MultiDiscrete(self._get_observation_space_list())
        self.view_len = view_len

        # self.observation_space = Tuple([Box(low=0.0, high=255.0, shape=(2 * self.view_len + 1,
        #                                                                 2 * self.view_len + 1, 3), dtype=np.float32),
        #                                 MultiDiscrete(self._get_observation_space_list())])
        self.vector_observation_dims = self._get_observation_space_list()
        self.max_dim = 1
        for dim in self.vector_observation_dims:
            self.max_dim *= dim

        self.observation_space = MultiDiscrete(self.vector_observation_dims) if observation_type == 'symbolic' else Box(
            low=0, high=255, shape=(self.view_len, self.view_len))
        self.observation_type = observation_type
        # self.observation_space = gym.spaces.Discrete(self.max_dim)

        self.bounded = False

        self.last_action = None

        self._seed()
        self.state = None
        self.dones = {taxi_name: False for taxi_name in self.taxis_names}
        self.dones['__all__'] = False

        self.np_random = None
        self.reset()

    def _get_observation_space_list(self) -> list:
        """
        Returns a list that emebed the observation space size in each dimension.
        An observation is a list of the form:
        [
            taxi_row, taxi_col,
            taxi_fuel,
            passenger1_row, passenger1_col,
            ...
            passenger_n_row, passenger_n_col,
            passenger1_dest_row, passenger1_dest_col,
            ...
            passenger_n_dest_row, passenger_n_dest_col,
            passenger1_status,
            ...
            passenger_n_status
        ]
        Returns: a list with all the dimensions sizes of the above.

        """
        locations_sizes = [self.num_rows, self.num_columns]
        fuel_size = [max(self.max_fuel) + 1]
        passengers_status_size = [self.num_taxis + 3]
        dimensions_sizes = []

        # for _ in range(self.num_taxis):
        dimensions_sizes += locations_sizes
        if self.can_see_others:
            dimensions_sizes += locations_sizes * (self.num_taxis - 1)
        # for _ in range(self.num_taxis):
        dimensions_sizes += fuel_size

        for _ in range(self.num_passengers):
            dimensions_sizes += 2 * locations_sizes
        # dimensions_sizes += 2 * locations_sizes
        for _ in range(self.num_passengers):
            dimensions_sizes += passengers_status_size
        # dimensions_sizes += passengers_status_size

        return dimensions_sizes

    # When running trials and finishing exploration - set seed to None
    def _seed(self, seed: int = 42) -> np.ndarray:
        """
        Setting a seed for the random sample state generation.
        Args:list
            seed: seed to use

        Returns: list[seed]

        """
        self.np_random, self.seed_id = seeding.np_random(seed)
        return np.array([self.seed_id])

    def reset(self, coordinates=None) -> dict:
        """
        Reset the environment's state:
            - taxis coordinates.
            - refuel all taxis
            - random get destinations.
            - random locate passengers.
            - preserve other definitions of the environment (collision, capacity...)
            - all engines turn on.
        Args:

        Returns: The reset state (dictionary as in RLlib format).

        """
        coordinates = np.array([[i, j] for i in range(self.num_rows) for j in range(self.num_columns)])
        coordinates = basic_utils.get_array_as_list_type(coordinates)
        self.current_step = 0
        # reset taxis locations
        taxis_locations = np.array(self.coordinates)[
                          np.random.choice(range(
                              len(self.coordinates) if isinstance(self.coordinates, list) else self.coordinates.shape[
                                  0]),
                              self.num_taxis, replace=False), :]
        taxis_locations = basic_utils.get_array_as_list_type(taxis_locations)
        self.collided = np.zeros(self.num_taxis)
        self.bounded = False
        self.window_size = self.view_len
        self.counter = 0

        # refuel everybody
        fuels = [self.max_fuel[i] for i in range(self.num_taxis)]

        # reset passengers
        self.coordinates = np.array(self.coordinates)
        self.coordinates = np.array(coordinates)
        self.passengers_locations = self.coordinates[
                                    np.random.choice(self.coordinates.shape[0], size=2 * self.num_passengers,
                                                     replace=False), :]
        # self.passengers_locations = self.coordinates

        chosen_indices = np.random.choice(self.passengers_locations.shape[0], size=self.num_passengers,
                                          replace=False)

        passengers_start_location = self.passengers_locations[chosen_indices, :]
        passengers_destinations = self.passengers_locations[
                                  list(set(range(self.passengers_locations.shape[0])) - set(list(chosen_indices))), :]
        # passengers_start_location = self.coordinates[: self.num_passengers]
        # passengers_destinations = self.coordinates[self.num_passengers: 2 * self.num_passengers]
        # passengers_start_location = self.passengers_locations[0, :]
        # passengers_destinations = self.passengers_locations[1, :]

        self.coordinates = basic_utils.get_array_as_list_type(self.coordinates)
        passengers_start_location = basic_utils.get_array_as_list_type(passengers_start_location)
        passengers_destinations = basic_utils.get_array_as_list_type(passengers_destinations)
        self.passengers_locations = basic_utils.get_array_as_list_type(self.passengers_locations)

        # Status of each passenger: delivered (1), in_taxi (positive number>2), waiting (2)
        passengers_status = [2 for _ in range(self.num_passengers)]
        self.state = [taxis_locations, fuels, passengers_start_location, passengers_destinations, passengers_status]

        self.last_action = None
        # Turning all engines on
        self.engine_status_list = list(np.ones(self.num_taxis))

        # resetting dones
        self.dones = {taxi_id: False for taxi_id in self.taxis_names}
        self.dones['__all__'] = False

        # setting observation according to the reset state and the rllib dictionary format
        observations = {}
        for i, taxi_id in enumerate(self.taxis_names):
            # observations[taxi_id] = self.change_observation_from_vector_to_number(observation_utils.get_status_vector_observation(self.state, taxi_id,
            #                                                                         self.taxis_names, self.num_taxis))
            if self.observation_type == 'symbolic':
                observations[taxi_id] = observation_utils.get_status_vector_observation(
                    self.state, taxi_id, self.taxis_names, self.num_taxis, self.can_see_others)
            else:
                observations[taxi_id] = observation_utils.get_image_obs_by_agent_id(agent_id=i, state=self.state,
                                                                                    num_taxis=self.num_taxis,
                                                                                    collided=self.collided,
                                                                                    view_len=self.view_len,
                                                                                    domain_map=self.desc)
        return observations

    def _add_custom_dropoff_for_passengers(self) -> (list, list):
        """
        Change the basic action list from config to have dropoff action for each specific passenger.
        for example, if we have 2 passengers, we'll have 2 seperate actions: "dropoff0", "dropoff1".
        Returns: adjusted ALL_ACTIONS_NAMES, BASE_AVAILABLE_ACTIONS lists

        """

        def add_custom_dropoff(list_of_actions: list) -> list:
            """
            A sub method to add the specific new actions (dropoffs) to a given action list.
            Args:
                list_of_actions: the action list to add on the new actions

            Returns: modified action list

            """
            temp_list = []
            for action in list_of_actions:
                if action != "dropoff":
                    temp_list.append(action)
                else:
                    for passenger_id in range(self.num_passengers):
                        temp_list.append(action + str(passenger_id))

            return temp_list

        new_base_action_list, new_all_action_list = add_custom_dropoff(BASE_AVAILABLE_ACTIONS), \
                                                    add_custom_dropoff(ALL_ACTIONS_NAMES)

        return new_all_action_list, new_base_action_list

    def _set_available_actions_dictionary(self) -> (list, dict, dict):
        """

        Generates list of all available actions in the parametrized domain, index->action dictionary to decode.
        Generation is based on the hyper-parameters passed to __init__ + parameters defined in control_config.py

        Returns: list of available actions, index->action dictionary for all actions and the reversed dictionary
        (action -> index).

        """
        ALL_ACTIONS_NAMES, BASE_AVAILABLE_ACTIONS = self._add_custom_dropoff_for_passengers()
        action_names = ALL_ACTIONS_NAMES  # From control_config.py
        base_dictionary = {}  # Total dictionary{index -> action_name}
        for index, action in enumerate(action_names):
            base_dictionary[index] = action

        available_action_list = BASE_AVAILABLE_ACTIONS  # From control_config.py

        if self.option_to_standby:
            available_action_list += ['turn_engine_on', 'turn_engine_off', 'standby']

        if not self.max_fuel[0] == 0:
            available_action_list.append('refuel')

        action_index_dictionary = dict((value, key) for key, value in base_dictionary.items())  # {action -> index} all
        available_actions_indexes = [action_index_dictionary[action] for action in available_action_list]
        index_action_dictionary = dict((key, value) for key, value in base_dictionary.items())

        return list(set(available_actions_indexes)), index_action_dictionary, action_index_dictionary

    def get_available_actions_dictionary(self) -> (list, dict):
        """
        Returns: list of available actions and index->action dictionary for all actions.

        """
        return self.available_actions_indexes, self.index_action_dictionary

    def step(self, action_dict: dict) -> (dict, dict, dict, dict):
        """
        Executing a list of actions (action for each taxi) at the domain current state.
        Supports not-joined actions, just pass 1 element instead of list.

        Args:
            action_dict: {taxi_name: action} - action of specific taxis to take on the step

        Returns: - dict{taxi_id: observation}, dict{taxi_id: reward}, dict{taxi_id: done}, _
        """

        rewards = {}
        self.current_step += 1

        randomized_order_of_taxis = list(action_dict.keys())
        random.shuffle(randomized_order_of_taxis)
        # Main of the function, for each taxi-i act on action[i]
        for taxi_name in randomized_order_of_taxis:
            action_list = action_dict[taxi_name]

            # meta operations on the type of the action
            action_list = basic_utils.get_action_list(action_list)

            for action in action_list:
                taxi = self.taxis_names.index(taxi_name)
                reward = reward_utils.partial_closest_path_reward(self.state, 'step')  # Default reward
                moved = False  # Indicator variable for later use
                rewards[taxi_name] = reward
                # taxi locations: [i, j]
                # fuels: int
                # passengers_start_locations and destinations: [[i, j] ... [i, j]]
                # passengers_status: [[1, 2, taxi_index+2] ... [1, 2, taxi_index+2]], 1 - delivered
                taxis_locations, fuels, passengers_start_locations, destinations, passengers_status = self.state
                self.dones = get_done_dictionary(self.dones, passengers_status, fuels, self.collided,
                                                 self.is_infinite_fuel, self.taxis_names)
                # done = all(loc == 1 for loc in passengers_status)
                if self.dones['__all__']:
                    continue

                if self.dones[taxi_name]:
                    continue

                # If taxi is collided, it can't perform a step
                if self.collided[taxi] == 1:
                    rewards[taxi_name] = reward_utils.partial_closest_path_reward(self.state, 'collided')
                    self.dones[taxi_name] = True
                    continue

                # If the taxi is out of fuel, it can't perform a step
                if fuels[taxi] == 0 and not basic_utils.at_valid_fuel_station(taxi, taxis_locations, self.fuel_stations,
                                                                              self.desc.copy().tolist(),
                                                                              self.fuel_type_list) and not \
                        self.is_infinite_fuel:
                    if self.dones[taxi_name]:
                        continue
                    else:
                        rewards[taxi_name] = reward_utils.partial_closest_path_reward(self.state, 'bad_refuel')
                        self.dones[taxi_name] = True
                        continue

                if self.dones[taxi_name]:
                    rewards[taxi_name] = -1
                    continue

                taxi_location = taxis_locations[taxi]
                row, col = taxi_location

                fuel = fuels[taxi]
                is_taxi_engine_on = self.engine_status_list[taxi]
                _, index_action_dictionary = self.get_available_actions_dictionary()

                if not is_taxi_engine_on:  # Engine is off
                    # update reward according to standby/ turn-on/ unrelated + turn engine on if requsted
                    rewards[taxi_name], self.engine_status_list = \
                        actions_utils.engine_is_off_actions(self.state, index_action_dictionary[action], taxi,
                                                            reward_utils.partial_closest_path_reward,
                                                            self.engine_status_list)


                else:  # Engine is on
                    # Binding
                    if index_action_dictionary[action] == 'bind':
                        self.bounded = False
                        rewards[taxi_name] = reward_utils.partial_closest_path_reward(self.state, 'bind')

                    # Movement
                    if 'goto' in index_action_dictionary[action]:
                        passenger_index = int(index_action_dictionary[action][8:])
                        if 'src' in index_action_dictionary[action]:
                            destination = passengers_start_locations[passenger_index]
                        else:
                            destination = destinations[passenger_index]

                        moved, row, col = actions_utils.take_movement(index_action_dictionary[action], row, col,
                                                                      self.num_rows, self.num_columns, self.desc,
                                                                      destination, self.column_partitions)

                    elif index_action_dictionary[action] in ['south', 'north', 'east', 'west']:
                        moved, row, col = actions_utils.take_movement(index_action_dictionary[action], row, col,
                                                                      self.num_rows, self.num_columns, self.desc)

                    # Check for collisions
                    if self.collision_sensitive_domain and moved:
                        if self.collided[taxi] == 0:
                            rewards[taxi_name], moved, action, taxis_locations, self.collided = \
                                actions_utils.check_action_for_collision(self.state, taxi, taxis_locations, row, col,
                                                                         moved, action, reward, self.num_taxis,
                                                                         self.option_to_standby,
                                                                         self.action_index_dictionary, self.collided,
                                                                         reward_utils.partial_closest_path_reward)

                    # Pickup
                    elif index_action_dictionary[action] == 'pickup':
                        passengers_status, rewards[taxi_name] = actions_utils.make_pickup(self.state, taxi,
                                                                                          passengers_start_locations,
                                                                                          passengers_status,
                                                                                          taxi_location, reward,
                                                                                          self.taxis_capacity,
                                                                                          reward_utils.partial_closest_path_reward)

                    # Dropoff
                    elif 'dropoff' in index_action_dictionary[action]:
                        passenger_index = int(index_action_dictionary[action][7:])
                        passengers_status, passengers_start_locations, rewards[taxi_name] = actions_utils.make_dropoff(
                            self.state,
                            taxi,
                            passengers_start_locations,
                            passengers_status,
                            destinations,
                            taxi_location,
                            reward,
                            passenger_index,
                            reward_utils.partial_closest_path_reward)

                    # Turning engine off
                    elif index_action_dictionary[action] == 'turn_engine_off':
                        rewards[taxi_name] = reward_utils.partial_closest_path_reward(self.state, 'turn_engine_off')
                        self.engine_status_list[taxi] = 0

                    # Standby with engine on
                    elif index_action_dictionary[action] == 'standby':
                        rewards[taxi_name] = reward_utils.partial_closest_path_reward(self.state, 'standby_engine_on')

                # Here we have finished checking for action for taxi-i
                # Fuel consumption
                if moved:
                    rewards[taxi_name], fuels[taxi], taxis_locations = actions_utils.update_movement_wrt_fuel(
                        self.state, taxi,
                        taxis_locations,
                        row, col, reward, fuel, self.is_infinite_fuel,
                        reward_utils.partial_closest_path_reward)

                if not 'goto_src0' in list(self.action_index_dictionary.keys()):
                    if (not moved) and action in [self.action_index_dictionary[direction] for
                                                  direction in ['north', 'south', 'west', 'east']]:
                        rewards[taxi_name] = self.rewards_table['hit_wall']

                # taxi refuel
                if index_action_dictionary[action] == 'refuel':
                    rewards[taxi_name], fuels[taxi] = actions_utils.refuel_taxi(self.state, fuel, reward, taxi,
                                                                                taxis_locations,
                                                                                self.fuel_stations,
                                                                                self.desc.copy().tolist(),
                                                                                self.fuel_type_list,
                                                                                self.is_infinite_fuel,
                                                                                self.max_fuel,
                                                                                reward_utils.partial_closest_path_reward)
                self.state[-1] = passengers_status
                self.dones = get_done_dictionary(self.dones, passengers_status, fuels, self.collided,
                                                 self.is_infinite_fuel, self.taxis_names)

        obs = {}
        for i, taxi_id in enumerate(action_dict.keys()):
            # obs[taxi_id] = self.change_observation_from_vector_to_number(observation_utils.get_status_vector_observation(self.state, taxi_id, self.taxis_names,
            #                                                                self.num_taxis))
            # obs[taxi_id] = observation_utils.get_status_vector_observation(
            #     self.state, taxi_id, self.taxis_names, self.num_taxis)
            if self.observation_type == 'symbolic':
                obs[taxi_id] = observation_utils.get_status_vector_observation(
                    self.state, taxi_id, self.taxis_names, self.num_taxis, self.can_see_others)
            else:
                obs[taxi_id] = observation_utils.get_image_obs_by_agent_id(agent_id=i, state=self.state,
                                                                           num_taxis=self.num_taxis,
                                                                           collided=self.collided,
                                                                           view_len=self.view_len,
                                                                           domain_map=self.desc)
        return obs, \
               rewards, self.dones, {}

    def render(self):
        render(self.desc.copy(), self.state, self.num_taxis, self.collided, self.last_action,
               self.action_index_dictionary, self.dones)

    def change_observation_from_vector_to_number(self, observation):
        number_observation = self.max_dim
        dim_offset_so_far = 1
        for i, obs in enumerate(observation):
            number_observation -= dim_offset_so_far * (self.vector_observation_dims[i] - obs)
            dim_offset_so_far *= self.vector_observation_dims[i]

        return number_observation
