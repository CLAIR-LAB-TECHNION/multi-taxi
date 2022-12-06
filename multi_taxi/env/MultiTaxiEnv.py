import sys
import warnings
from collections import OrderedDict, Counter
from enum import Enum
from functools import lru_cache
from typing import Union, Dict, List

import numpy as np
from gym import spaces
from gym.utils import colorize, seeding
from pettingzoo import ParallelEnv

from . import config
from ..env.reward_tables import TAXI_ENVIRONMENT_REWARDS, PICKUP_ONLY_TAXI_ENVIRONMENT_REWARDS
from ..env.state import MultiTaxiEnvState
from ..utils import ansitoimg
from ..utils.stochastic_action_function import JointStochasticActionFunction
from ..utils.types import FuelType, ObservationType, Action, MOVE_ACTIONS, ENGINE_ACTIONS, Event, SymbolicObservation
from ..world.domain_map import DomainMap
from ..world.entities import Taxi, Passenger, PASSENGER_NOT_IN_TAXI
from ..world.maps import DEFAULT_MAP

PASSENGER_RENDERING_CHAR = 'P'
DESTINATION_RENDERING_CHAR = 'D'

PerTaxiValue = lambda t: Union[t, List[t]]


class MultiTaxiEnv(ParallelEnv):
    """
    The Taxi Problem
    From "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition" by Tom Dietterich.
    Adjusted for MARL by Ofir Abu.
    Debugged and adapted for PettingZoo by Guy Azran

    Description:
    A user-configured number of taxis and passengers are placed randomly within a configurable map, along with a random
    destination for each passenger. It is the taxis' objective to transport all passengers to their respective
    destinations. In order to achieve this objective, the taxis must navigate to the passenger locations, pick them up,
    navigate to the correct destination and drop the passenger off. Taxis may need to watch their fuel levels, passenger
    capacities, and many more parameters. They might also need to take collisions into consideration, forcing them to
    coordinate flawlessly or face their demise.
    This environment is highly configurable. It provides global configurations that affect the environment, rewards, and
    agent objectives. Furthermore, one may configure custom observation spaces, action spaces, and reward functions
    specified on a per-taxi level. The environment supports as many taxis as can fit in the given domain map, but taxi
    rendering colors may overlap due to ANSI text limitations. see the `__init__` function for a full list of
    configurations and their description.

    Domain Maps:
    environment domain maps are defined as lists of strings. below is an example of a correctly defined map:
    [
        "+-----------------------+",
        "| : |F: | : | : | : |G: |",
        "| : : : : : | : : : : : |",
        "|X:X:X:X: : : : :X:X:X:X|",
        "|X:X:X:X: : | : :X:X:X:X|",
        "|X:X:X:X: : | : :X:X:X:X|",
        "| : : : : : : : : : : : |",
        "| | :G| | | :F| | | : | |",
        "+-----------------------+",
    ]
    each string is a row of the map. the first and last rows and columns are barrier values (hyphen "-" and pipe "|"),
    with the corners defined as "+" characters. These define the boundary of the domain map. The inner values define the
    map itself. Free cells, where taxis and passengers can be located, are shown as empty (space) characters. between
    two horizontally adjacent cells there is either a wall (pipe "|") or a free transition (colon ":"). taxis can
    transition freely between vertically adjacent cells. Cells marked with blockades ("X" character) cannot be occupied.
    although maps must be rectangular, i.e., the number of columns in each row must be identical, we can define
    irregular shapes using blockades ("X" character) as seen in the above example. fuel stations are also defined in the
    domain map. a cell can contain either the value "F" for standard fuel stations, or "G" for the alternative gas
    stations. both types of fuel are identical, but taxis can only use one fuel type or the other, meaning they must go
    be at a station of the appropriate type in order to successfully refuel.

    Observations:
    There are two kinds of observations that taxis can receive: symbolic and image. The symbolic observation is a vector
    of symbolic values about the environment, e.g., taxi locations, fuel level, engine status, passenger info, and more.
    For a full list of symbolic values, see `multi_taxi.utils.types.SymbolicObservation`. The image observation is a
    pixel representation of a window around the agent within the domain map.

    Actions:
    Agent actions can be divided into several categories:
      - navigation - movement actions "north", "south", "east", "west"
      - passenger pickup/dropoff - taxis can pick up passengers at their locations and drop them off anywhere on the
                                   map. both pickup and dropoff operations can come in two forms: generic and specific.
                                     - specific pickup/dropoff: the action space contains pickup and dropoff actions
                                                                for each passenger individually. to perform the action
                                                                correctly, the relevant passenger must be in the right
                                                                position (at the taxi location for pickup or in the
                                                                taxi for dropoff).
                                     - generic pickup: the action space contains a single pickup action that will pick
                                                       up any passenger at the current taxi location. If multiple
                                                       passengers are at that location, one is picked up (the one with
                                                       the lowest ID).
                                     - generic dropoff: the action space contains a single dropoff action that will
                                                        drop off a passenger that is in the taxi whose destination
                                                        location is at the taxi location, i.e., will drop off the
                                                        most relevant passenger. If the taxi currently holds multiple
                                                        passengers whose destination is at the taxi's location, the
                                                        taxi will drop off the one with the lowest ID. If no passenger
                                                        destination is at the taxi's location, then the riding
                                                        passenger with the lowest ID will be dropped off.
                                   taxi pickup and dropoff action forms can be specified on a per-taxi level.
      - engine control - actions that change the taxi engine status (on / off). When the engine is off, fuel is not
                         consumed at each step, but some actions cannot be performed.
      - standby - "do nothing" action
      - refuel - allows refueling when the taxi is at a relevant fuel station.
    for the user's convenience, a `get_action_meanings` method is provided to show all the actions and a string
    disambiguation of each action after environment initialization.
    actions may be stochastic. the user can set the probability to perform a different actions than the intended. this
    is configured using the `stochastic_actions` environment argument.


    Rewards:
    an agent's rewards are determined by the `reward_table` environment argument. each taxi can have its own reward
    table with different reward value. Rewards are received during the environment step for the occurrence of certain
    events. A full list of events is available at `multi_taxi.utils.types.Event`. The default reward tables can be found
    in `multi_taxi.env.reward_tables`. Any reward that is not specified in the `reward_table` argument is taken from
    these default tables. note that there are multiple reward tables for different objectives, chosen accroding to the
    environment configuration (determined via the given arguments).

    Rendering:
    the environment is rendered as ANSI text. highlighted positions indicate taxi positions, colored "P" values indicate
    passenger locations and matching colored "D" values indicate the passengers' respective destinations. note that
    values may overlap, e.g. two passengers at the same location. Since it is not possible to render multiple characters
    at the same map position, some information may be missing in the rendering. to compensate for this, the environment
    status is also printed as text below the rendering.
    """

    ##################
    # Pettingzoo API #
    ##################

    # metadata object defining information about the environment
    metadata = {'render_modes': ['human', 'ansi'], 'name': 'multi_taxi_env_v0'}

    def __init__(self,
                 # environment configurations
                 num_taxis: int = None,
                 num_passengers: int = None,
                 domain_map: List[str] = None,
                 pickup_only: bool = None,
                 no_intermediate_dropoff: bool = None,
                 intermediate_dropoff_reward_by_distance: bool = None,
                 distinct_taxi_initial_locations: bool = None,
                 distinct_passenger_initial_pickups: bool = None,
                 distinct_passenger_dropoffs: bool = None,
                 allow_collided_taxis_on_reset: bool = None,
                 allow_arrived_passengers_on_reset: bool = None,
                 clear_dead_taxis: bool = None,

                 # taxi configurations
                 max_steps: PerTaxiValue(int) = None,
                 max_capacity: PerTaxiValue(int) = None,
                 max_fuel: PerTaxiValue(int) = None,
                 fuel_type: PerTaxiValue(FuelType) = None,
                 has_standby_action: PerTaxiValue(bool) = None,
                 has_engine_control: PerTaxiValue(bool) = None,
                 engine_off_on_empty_tank: PerTaxiValue(bool) = None,
                 can_refuel_without_fuel: PerTaxiValue(bool) = None,
                 can_collide: PerTaxiValue(bool) = None,
                 passenger_fumble: PerTaxiValue(bool) = None,
                 specify_passenger_pickup: PerTaxiValue(bool) = None,
                 specify_passenger_dropoff: PerTaxiValue(bool) = None,
                 reward_table: PerTaxiValue(Dict[Event, float]) = None,
                 stochastic_actions: PerTaxiValue(Dict[str, str]) = None,
                 observation_type: PerTaxiValue(ObservationType) = None,
                 can_see_other_taxi_info: PerTaxiValue(bool) = None,
                 field_of_view: PerTaxiValue(int) = None,

                 initial_seed=None):
        """
        Initialize environment with given specifications. Possible configurations are as follows:

        - Environment configurations: num_taxis, num_passengers, domain_map, pickup_only,
                                      intermediate_dropoff_reward_by_distance, distinct_taxi_initial_locations,
                                      distinct_passenger_initial_pickups, distinct_passenger_dropoffs,
                                      allow_collided_taxis_on_reset, allow_arrived_passengers_on_reset, clear_dead_taxis

        - Taxi configurations: max_capacity, max_fuel, fuel_type, has_standby_action, has_engine_control,
                               engine_off_on_empty_tank, can_collide, specify_passenger_pickup,
                               specify_passenger_dropoff, reward_table, stochastic_actions, observation_type,
                               can_see_other_taxi_info, field_of_view

        Environment configurations must be a specific type of value, while taxi configurations are either a specific
        type of value or a list of that type which is the same length as the number of taxis in the environment. Taxi
        configurations that are a single value will be used for all taxis. Any configuration, including specific taxi
        configurations (i.e., values within a given list of configurations) may be replaced with `None` to use the
        default configuration. Configuration defaults can be found in `multi_taxi.env.config`.

        Args:
            num_taxis: the number of taxis operating in the environment.
            num_passengers: the number of passengers to drive in the environment.
            domain_map: array of strings representing the environment map with special characters for taxis initialized
                        spots and fuel stations(see `multi_taxi.world.maps.DEFAULT_MAP`).
            pickup_only: simplifies the problem to only pick up all passengers, without needing dropping them off.
            no_intermediate_dropoff: if True, taxis an only drop passengers off at their destination.
            intermediate_dropoff_reward_by_distance: changes the reward function for dropping off passengers at a
                                                     location that is not their final destination. if `True`, the given
                                                     reward for intermediate dropoffs is the negative Manhattan distance
                                                     from the dropoff location and the passenger's true destination.
                                                     otherwise, the given reward is that of the taxi's reward table.
            distinct_taxi_initial_locations: asserts distinct start locations for taxis on reset.
            distinct_passenger_initial_pickups: asserts distinct start locations for passengers on reset.
            distinct_passenger_dropoffs: asserts distinct passenger destinations on reset.
            allow_collided_taxis_on_reset: if `True`, taxis may start at the same location and are considered to be
                                           collided.
            allow_arrived_passengers_on_reset: if `True`, passengers may start at the same location as their destination
                                               and are considered to have arrived.
            clear_dead_taxis: if `True`, taxis that can no longer act (i.e. dead) are completely removed from the
                              environment. otherwise, dead taxis remain in the environment, continue receiving rewards,
                              and can be collided into.
            max_steps: determines the maximum number of actions a taxi can take at each reset.
            max_capacity: determines a taxi's maximum passenger capacity.
            max_fuel: determines a taxi's maximum fuel capacity
            fuel_type: determines a taxi's required fuel station for refueling
            has_standby_action: if `True`, the taxi has the ability to perform the "standby" action, i.e., do nothing.
            has_engine_control: if `True`, the taxi can turn the engine on and off.
            engine_off_on_empty_tank: if `True`, if the taxi runs out of fuel and does not refuel at that step, the
                                      taxi's engine will turn off. This parameter is ignored if the taxi does not have
                                      engine control.
            can_refuel_without_fuel: if `False`, a taxi is considered dead when its fuel capacity reaches 0. if `True`,
                                     a taxi with 0 fuel may still act if it is on a valid fuel station.
            can_collide: if `True`, the taxi becomes a collidable and may collide with other collidable taxis.
            passenger_fumble: if `True`, all carried passengers are dropped off when the taxi dies at the location
                              of the taxi's death.
            specify_passenger_pickup: if `True`, the taxi's "pickup" actions must indicate the exact passenger they
                                      intend to pick up. otherwise, a generic pickup action is used. generic pickup
                                      will pick up any passenger at the taxi's current location. if multiple passengers
                                      meet this criterion, the passenger with the lowest ID is picked up.
            specify_passenger_dropoff: if `True`, the taxi's "dropoff" actions must indicate the exact passenger they
                                       intend to drop off. otherwise, a generic dropoff action is used. generic dropoff
                                       will drop off the passenger whose destination is at the taxi's current location.
                                       if multiple passengers meet this criterion, the passenger with the lowest ID
                                       is dropped off. if no passenger meets this criterion, the passenger in the taxi
                                       with the lowest ID is dropped off.
            reward_table: a dictionary that describes the taxi's reward function upon the occurrence of certain events
                          (see `multi_taxi.utils.types.Event`).
            stochastic_actions: a dictionary that describes a taxi's probabilities to perform different actions than the
                                intended one. for example, using the below dictionary as input will make all taxis
                                perform the "west" action at 25% probability when choosing the "north" action:
                                {
                                    'north': {
                                                'north': 0.75,
                                                'west': 0.25
                                             }
                                }.
                                see `multi_taxi.utils.types.Action` for available actions. see
                                `multi_taxi.utils.stochastic_actions` for more details.
            observation_type: determines the type of observations received for the taxi. There are three types of
                              observations:
                                  SYMBOLIC - A vector of symbolic values
                                  IMAGE - An RGB image of a window surrounding the taxi.
                                  MIXED -  A dictionary of symbolic and image observations
                              see `multi_taxi.utils.types.ObservationType`. Use the `get_observation_meanings` API
                              method for deeper insight into the taxi's observations.
            can_see_other_taxi_info: adds information for other taxis into symbolic observations. ignored when image
                                     observations are used.
            field_of_view: defines the dimension of the square window around the taxi within the domain map for image
                           observations. if not specified, the image observation is the entire map.
            initial_seed: sets deterministic randomness in the environment.
        """
        # set random seed if specified
        self.__np_random = None
        self.seed(initial_seed)

        # initialize env configurations with argument or default value
        self.num_taxis = self.__single_value_config(num_taxis, int, config.DEFAULT_NUM_TAXIS)
        self.num_passengers = self.__single_value_config(num_passengers, int, config.DEFAULT_NUM_PASSENGERS)
        self.pickup_only = self.__single_value_config(pickup_only, bool, config.DEFAULT_PICKUP_ONLY)
        self.no_intermediate_dropoff = self.__single_value_config(
            no_intermediate_dropoff, bool, config.DEFAULT_NO_INTERMEDIATE_DROPOFF
        )
        self.intermediate_dropoff_penalty_by_distance = self.__single_value_config(
            intermediate_dropoff_reward_by_distance, bool, config.DEFAULT_INTERMEDIATE_DROPOFF_REWARD_BY_DISTANCE
        )
        self.distinct_taxi_initial_locations = self.__single_value_config(
            distinct_taxi_initial_locations, bool, config.DEFAULT_DISTINCT_TAXI_INITIAL_LOCATIONS
        )
        self.distinct_passenger_initial_pickups = self.__single_value_config(
            distinct_passenger_initial_pickups, bool, config.DEFAULT_DISTINCT_PASSENGER_INITIAL_PICKUPS
        )
        self.distinct_passenger_dropoffs = self.__single_value_config(
            distinct_passenger_dropoffs, bool, config.DEFAULT_DISTINCT_PASSENGER_DROPOFFS
        )
        self.allow_arrived_passengers_on_reset = self.__single_value_config(
            allow_arrived_passengers_on_reset, bool, config.DEFAULT_ALLOW_ARRIVED_PASSENGERS_ON_RESET
        )
        self.allow_collided_taxis_on_reset = self.__single_value_config(
            allow_collided_taxis_on_reset, bool, config.DEFAULT_ALLOW_COLLIDED_TAXIS_ON_RESET
        )
        self.clear_dead_taxis = self.__single_value_config(clear_dead_taxis, bool, config.DEFAULT_CLEAR_DEAD_TAXIS)

        # setup domain map object
        self.domain_map = DomainMap(domain_map or DEFAULT_MAP)

        # pettingzoo required attributes
        self.possible_agents = [f'taxi_{i}' for i in range(self.num_taxis)]
        self.agents = []

        # initialize taxi configurations with argument or default value
        self.max_steps = self.__per_taxi_single_or_list(max_steps, int, config.DEFAULT_MAX_STEPS)
        if not pickup_only:
            self.max_capacity = self.__per_taxi_single_or_list(max_capacity, int, config.DEFAULT_MAX_CAPACITY)
        else:
            self.max_capacity = self.__per_taxi_single_or_list(None, int, config.DEFAULT_MAX_CAPACITY)
        self.max_fuel = self.__per_taxi_single_or_list(max_fuel, int, config.DEFAULT_MAX_FUEL)
        self.fuel_type = self.__per_taxi_single_or_list(fuel_type, FuelType, config.DEFAULT_FUEL_TYPE)
        self.has_standby_action = self.__per_taxi_single_or_list(has_standby_action, bool,
                                                                 config.DEFAULT_CAN_STANDBY)
        self.has_engine_control = self.__per_taxi_single_or_list(has_engine_control, bool,
                                                                 config.DEFAULT_HAS_ENGINE_CONTROL)
        self.engine_off_on_empty_tank = self.__per_taxi_single_or_list(engine_off_on_empty_tank, bool,
                                                                       config.DEFAULT_ENGINE_OFF_ON_EMPTY_TANK)
        self.engine_off_on_empty_tank = {
            # only allow this setting for taxis that have engine control
            k: v and self.has_engine_control[k] for k, v in self.engine_off_on_empty_tank.items()
        }
        self.can_refuel_without_fuel = self.__per_taxi_single_or_list(can_refuel_without_fuel, bool,
                                                                      config.DEFAULT_CAN_REFUEL_WITHOUT_FUEL)
        self.can_collide = self.__per_taxi_single_or_list(can_collide, bool, config.DEFAULT_CAN_COLLIDE)

        # if only one taxi can collide, then collisions are not possible
        collision_possible = sum(int(v) for v in self.can_collide.values()) > 1
        self.can_collide = {
            # only allow "can_collide" configuration if collisions are possible
            k: v and collision_possible for k, v in self.can_collide.items()
        }

        self.passenger_fumble = self.__per_taxi_single_or_list(passenger_fumble, bool, config.DEFAULT_PASSENGER_FUMBLE)
        self.specify_passenger_pickup = self.__per_taxi_single_or_list(specify_passenger_pickup, bool,
                                                                       config.DEFAULT_SPECIFY_PASSENGER_PICKUP)
        self.specify_passenger_dropoff = self.__per_taxi_single_or_list(specify_passenger_dropoff, bool,
                                                                        config.DEFAULT_SPECIFY_PASSENGER_DROPOFF)

        # default rewards table set according to "pickup only" setting
        default_rt = PICKUP_ONLY_TAXI_ENVIRONMENT_REWARDS.copy() if pickup_only else TAXI_ENVIRONMENT_REWARDS.copy()

        # set all custom reward tables for each agent
        custom_rt = self.__per_taxi_single_or_list(reward_table, dict, default_rt)

        # set final reward table for each agent as the default reward table updated by the custom rewards.
        # this removes the need to check for redundant keys because we can just ignore them.
        self.reward_table = {agent: default_rt.copy() for agent in self.possible_agents}
        [self.reward_table[agent].update(agent_rt) for agent, agent_rt in custom_rt.items()]

        self.stochastic_action_function = JointStochasticActionFunction(
            probs_dict_dict=self.__per_taxi_single_or_list(stochastic_actions, dict, {})
        )

        # allow string values
        self.observation_type = self.__per_taxi_single_or_list(observation_type, ObservationType,
                                                               config.DEFAULT_OBSERVATION_TYPE)
        if any(obs_type in [ObservationType.IMAGE, ObservationType.MIXED]
               for obs_type in self.observation_type.values()):
            self.__image_render_helper = ansitoimg.TaxiMapRendering(self.domain_map.domain_map)
        else:
            self.__image_render_helper = None

        # initialize observation type specific configurations
        self.can_see_other_taxi_info = {  # other taxis info used only if symbolic observation is required
            k: is_aware if self.observation_type[k] in [ObservationType.SYMBOLIC, ObservationType.MIXED] else False
            for k, is_aware in self.__per_taxi_single_or_list(can_see_other_taxi_info, bool,
                                                              config.DEFAULT_CAN_SEE_OTHER_TAXI_INFO).items()
        }
        self.field_of_view = {  # FOV used only if image observation is required
            k: fov if self.observation_type[k] in [ObservationType.IMAGE, ObservationType.MIXED] else 0
            for k, fov in self.__per_taxi_single_or_list(field_of_view, int, config.DEFAULT_FIELD_OF_VIEW).items()
        }

        # useful value to check if fuel and timestep considerations are required
        self.infinite_fuel = {taxi: f == float('inf') for taxi, f in self.max_fuel.items()}
        self.infinite_steps = {taxi: s == float('inf') for taxi, s in self.max_steps.items()}

        # useful for checking if taxis must be sampled distinctly.
        # reset locations should be sampled distinctly if specified by the user or if all taxis can collide and
        # collisions are not allowed on reset
        self.sample_locations_distinctly = (self.distinct_taxi_initial_locations or
                                            (all(self.can_collide.values()) and not self.allow_collided_taxis_on_reset))

        # initialize observation and action spaces
        self.__observation_spaces, self.__observation_space_meanings = self.__get_observation_spaces_and_meanings()
        self.__action_index_to_name = self.__get_action_space_disambiguation()
        self.__action_name_to_index = {taxi: {name: idx for idx, name in self.__action_index_to_name[taxi].items()}
                                       for taxi in self.__action_index_to_name}  # flip index-name to name-index pairs

        # state and env objects
        self.__state = None

    def reset(self, seed=None):
        # set seed if given
        if seed is not None:
            self.seed(seed)

        # reset agents
        self.agents = self.possible_agents.copy()

        # initialize a random state
        taxis = self.__random_taxis()
        passengers = self.__random_passengers()
        self.__state = MultiTaxiEnvState(taxis, passengers)

        # return all observations
        return self.__observe_all()

    def seed(self, seed=None):
        self.__np_random, _ = seeding.np_random(seed)

    def step(self, actions: dict):

        # use string actions for readability
        actions = {agent: self.__action_index_to_name[agent][action] for agent, action in actions.items()}

        # sample from stochastic joint actions space
        true_actions = self.stochastic_action_function(actions, rng=self.__np_random)

        # step in environment with sampled action
        new_state, rewards, dones, infos = self.__step_from_state(self.__state, true_actions)

        # log desired actions and performed transitions
        for agent, agent_info in infos.items():
            agent_info['desired_action'] = actions[agent]
            agent_info['performed_transition'] = true_actions[agent]

        # set new state in environment
        self.__state = new_state

        # get new observations according to the current state
        obs = self.__observe_all()

        # remove done agents from live agents list
        for taxi_name, done in dones.items():
            if done:
                self.agents.remove(taxi_name)

        return obs, rewards, dones, infos

    def render(self, mode='human'):
        rendering = self.__render_map()  # get map string

        rendering += '\n'  # separate map and text with new line

        rendering += self.__render_status()  # get state info string

        # return string value for ansi mode
        if mode == 'ansi':
            return rendering
        else:
            print(rendering)
            sys.stdout.flush()

    def state(self):
        return self.__state.copy()

    @lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.__observation_spaces[agent]

    @lru_cache(maxsize=None)
    def action_space(self, agent):
        n = len(self.__action_index_to_name[agent])
        return spaces.Discrete(n)

    ######################
    # End Pettingzoo API #
    ######################

    #######################
    # Extra API Functions #
    #######################

    def state_action_transitions(self, state: MultiTaxiEnvState, actions: dict):
        """
        performs a transition on the given state with a given set of actions. this method does not change the
        environment state. to do so, use the `set_state` method with on of the returned state.

        Args:
            state: the state from which to transition
            actions: the joint action to perform for the transition

        Returns:
            a list of tuples (new_state, rewards, dones, infos, prob) that each represent a possible transition within
            the environment from the given state when performing the given actions.
                - new_state: the new state reached after the transition
                - rewards: a dictionary of rewards received by each taxi during th transition
                - infos: a dictionary of extra info about the performed transition
                - dones: a dictionary indicating which agents are "done" after the performed transition.
        """
        # use string actions for readability
        actions = {agent: self.__action_index_to_name[agent][action] for agent, action in actions.items()}

        # results container
        transition_results = []

        # iterate all possible joint actions
        for joint_action, prob in self.stochastic_action_function.iterate_possible_actions(actions):
            # make a copy of the current state. update only the copy
            new_state, rewards, dones, infos = self.__step_from_state(state, joint_action)

            # log desired actions and performed transitions
            for agent, agent_info in infos.items():
                agent_info['desired_action'] = actions[agent]
                agent_info['performed_transition'] = joint_action[agent]

            # append results to returned container
            transition_results.append((new_state, rewards, dones, infos, prob))

        return transition_results

    def set_state(self, state: MultiTaxiEnvState):
        """
        Sets the current environment state

        Args:
            state: the state to set in the environment
        """
        self.__state = state

    def observe_all(self):
        """
        gets all agent observations for the given state.
        This is an API exposure of an inner method.

        Returns:
            a dictionary for all agent observations.
        """
        return self.__observe_all()

    def get_observation_meanings(self, agent):
        """
        provides an informative output for understanding the observation space.

        Args:
            agent: the name of the agent for which to get the observation meanings

        Returns:
            an informative data structure that demystifies the observation space for the given `agent`
        """
        return self.__observation_space_meanings[agent]

    def get_action_meanings(self, agent):
        """
        provides an informative output for understanding the action space

        Args:
            agent: the name of the agent for which to get the action meanings

        Returns:
            an informative data structure that demystifies the action space for the given `agent`
        """
        return self.__action_index_to_name[agent].copy()

    def get_action_map(self, agent):
        """
        provides a mapping between action names and their respective index

        Args:
            agent: the name of the agent for which to get the action map

        Returns:
            A dictionary that maps action names to their discrete space index
        """
        return self.__action_name_to_index[agent].copy()

    def env_done(self, state=None):
        """
        tells the caller whether the environment run is complete and must be reset. An environment is done if either all
        taxis are dead or if the environment objective has been reached.

        a taxi is considered dead before the end of the episode if one of the following holds:
            1. taxi has collided.
            2. taxi has run out of fuel and is not in a correct fuel station.

        the objective has been reached if one of the following holds:
            1. `pickup_only` is set to `False` and all passengers have arrived at their destinations.
            2. `pickup_only` is set to `True` and all passengers are in a taxi.

        Args:
            state: the state in which we want to check the "done" status

        Returns:
            `True` if the environment is done, `False` otherwise.
        """
        if state is None:
            state = self.__state

        # env is done if all taxis are done
        return all(self.__taxi_is_dead(taxi) for taxi in state.taxis) or self.__objective_achieved(state)

    ###########################
    # End Extra API Functions #
    ###########################

    ####################################################################################################################

    ############################
    # Initialization Functions #
    ############################

    def __per_taxi_single_or_list(self, value, value_type, default_value):
        if isinstance(value, list):
            # list input
            # give specific configuration to each taxi.
            # can use None to get default value or specific taxi

            # assert list length matches number of agents
            assert len(self.possible_agents) == len(value), f'per taxi cfg {value} must contain a value for each taxi'

            # check each value as standalone for each taxi
            clean_values = [self.__single_value_config(v, value_type, default_value) for v in value]

            # map agent names to list values
            return OrderedDict(zip(self.possible_agents, clean_values))
        else:
            # check as single value for all taxis
            checked_value = self.__single_value_config(value, value_type, default_value)
            return OrderedDict((taxi, checked_value) for taxi in self.possible_agents)

    @staticmethod
    def __single_value_config(value, value_type, default_value):
        # correct value types should be returned as is
        # correct <==> of the specified value type or equal to the default value
        if isinstance(value, value_type) or value == default_value:
            return value

        # allow value input for Enum types
        if issubclass(value_type, Enum):
            try:
                # if possible, override default value with given input
                default_value = value_type(value)
                value = None  # suppress later warning
            except ValueError:
                # value is not a member of the enum. cannot create enum value
                pass

        # if given value is not None, then a bad input was given. revert to default but give warning.
        # if None was given, use default without warning.
        if value is not None:
            warnings.warn(f'bad value type {type(value)} given as config in {value}. expected type {value_type}')

        # map agent names to the default value
        return default_value

    def __random_taxis(self):
        # sample initial taxi locations
        start_locations = self.domain_map.sample_free_locations(self.num_taxis,
                                                                distinct=self.sample_locations_distinctly,
                                                                rng=self.__np_random)

        # check for collisions on initial locations if not sampled discretely and collisions are not allowed
        if not self.sample_locations_distinctly and not self.allow_collided_taxis_on_reset:
            start_locations = self.__fix_colliding_taxis(start_locations)
            collided_list = [False] * self.num_taxis
        else:
            # collisions can occur on reset. check for any location overlaps
            collided_list = [any(np.all(loc1 == loc2) and i != j for j, loc2 in enumerate(start_locations))
                             for i, loc1 in enumerate(start_locations)]

        taxis = Taxi.from_lists([int(agent.split('_')[-1]) for agent in self.possible_agents],  # taxi enumeration as ID
                                start_locations,  # a random starting location for each taxi
                                list(self.max_capacity.values()),
                                list(self.max_fuel.values()),  # all taxis start with max fuel
                                list(self.max_fuel.values()),
                                list(self.fuel_type.values()),
                                [0] * self.num_taxis,  # all taxis start at step 0
                                list(self.max_steps.values()),
                                [None] * self.num_taxis,  # no taxi is carrying a passenger
                                list(self.can_collide.values()),
                                collided_list,
                                [True] * self.num_taxis)  # all taxi engines are on

        # check for collided taxis

        return taxis

    def __fix_colliding_taxis(self, start_locations):
        colliding_taxis = [i for i, can_collide in enumerate(self.can_collide.values()) if can_collide]
        if len(colliding_taxis) > 1:
            colliding_taxis_start_locations = self.domain_map.sample_free_locations(len(colliding_taxis),
                                                                                    distinct=True,
                                                                                    rng=self.__np_random)
            for i, loc in zip(colliding_taxis, colliding_taxis_start_locations):
                start_locations[i] = loc

        return start_locations

    def __random_passengers(self):
        # sample passenger initial locations and dropoff destinations
        start_locations = self.domain_map.sample_free_locations(self.num_passengers,
                                                                distinct=self.distinct_passenger_initial_pickups,
                                                                rng=self.__np_random)

        if self.pickup_only:
            # give invalid (and unreachable) destination
            dest_locations = [(-1, -1) for _ in range(self.num_passengers)]
        else:
            dest_locations = self.domain_map.sample_free_locations(self.num_passengers,
                                                                   distinct=self.distinct_passenger_dropoffs,
                                                                   rng=self.__np_random)

            if not self.allow_arrived_passengers_on_reset:
                start_locations, dest_locations = self.__fix_passenger_at_destination(start_locations, dest_locations)

        return Passenger.from_list(range(self.num_passengers), start_locations, dest_locations,
                                   [PASSENGER_NOT_IN_TAXI] * self.num_passengers)

    def __fix_passenger_at_destination(self, start_locations, dest_locations):
        # check that all passenger start and dest are different
        if self.num_passengers == 1 and np.all(start_locations == dest_locations):  # single passenger, keep sampling
            while np.all(start_locations == dest_locations):
                dest_locations = self.domain_map.sample_free_locations(self.num_passengers, rng=self.__np_random)

            return start_locations, dest_locations
        else:  # multiple passengers, swap distinct destination with some other one
            duplicate_locations_passenger = []
            for i, (s, d) in enumerate(zip(start_locations, dest_locations)):
                if np.all(s == d):
                    duplicate_locations_passenger.append(i)

            for i in duplicate_locations_passenger:  # swap duplicate dest with some other destination
                if self.distinct_passenger_dropoffs:
                    if np.all(start_locations[i] == dest_locations[i]):  # check that it wasn't already swapped
                        tmp = dest_locations[i - 1]
                        dest_locations[i - 1] = dest_locations[i]
                        dest_locations[i] = tmp
                else:
                    while np.all(start_locations[i] == dest_locations[i]):
                        dest_locations[i] = self.domain_map.sample_free_locations(n=None, rng=self.__np_random)

        return start_locations, dest_locations

    ################################
    # End Initialization Functions #
    ################################

    ##################
    # Step Functions #
    ##################

    def __step_from_state(self, state, joint_action):
        new_state = state.copy()
        rewards = {taxi_name: 0 for taxi_name in self.agents}
        infos = {taxi_name: {'events': []} for taxi_name in self.agents}

        # do step within new state
        for taxi_name, action in joint_action.items():
            taxi = new_state.taxi_by_name[taxi_name]

            # check if taxi can step at all
            if self.__taxi_is_dead(taxi):
                self.__add_reward(taxi.name, infos, rewards, Event.DEAD)
                infos[taxi.name]['dead'] = True
                continue

            # step will execute
            infos[taxi.name]['dead'] = False
            self.__add_reward(taxi.name, infos, rewards, Event.STEP)
            taxi.n_steps += 1

            # switch on actions
            if action in {a.value for a in MOVE_ACTIONS}:
                self.__move_taxi(taxi, action, infos, rewards)
            elif action.startswith(Action.PICKUP.value):
                if self.specify_passenger_pickup[taxi.name]:
                    self.__specific_pickup(taxi, new_state.passengers, action, infos, rewards)
                else:
                    self.__generic_pickup(taxi, new_state.passengers, infos, rewards)
            elif action.startswith(Action.DROPOFF.value):
                if self.specify_passenger_dropoff[taxi.name]:
                    self.__specific_dropoff(taxi, action, infos, rewards)
                else:
                    self.__generic_dropoff(taxi, infos, rewards)
            elif action == Action.STANDBY.value:
                self.__taxi_standby(taxi, infos, rewards)
            elif action == Action.ENGINE_ON.value:
                self.__taxi_engine_on(taxi, infos, rewards)
            elif action == Action.ENGINE_OFF.value:
                self.__taxi_engine_off(taxi, infos, rewards)
            elif action == Action.REFUEL.value:
                self.__taxi_refuel(taxi, infos, rewards)

            # each step with the engine one costs 1 fuel unit
            # only reduce fuel if the taxi has fuel, didn't refuel at this step and the engine is on
            if not taxi.empty_tank and taxi.engine_on and not (action == Action.REFUEL.value and
                                                               infos[taxi.name]['refuel_success']):
                taxi.fuel -= 1

            if taxi.engine_on and taxi.empty_tank and self.engine_off_on_empty_tank[taxi.name]:
                taxi.engine_on = False

        self.__check_collisions(state, new_state, infos, rewards)

        if self.__objective_achieved(new_state):
            self.__objective_achieved_reward(new_state, infos, rewards)
        else:
            # it is ok to be stuck without fuel or run out of time if the objective is achieved at that step
            self.__check_stuck_without_fuel(new_state, infos, rewards)
            self.__check_out_of_time(new_state, infos, rewards)


        # drop all passengers of dead taxis if configured to do so
        for taxi_name in self.agents:
            if self.passenger_fumble[taxi_name] and self.__taxi_is_dead(taxi_name):
                taxi = new_state.taxi_by_name[taxi_name]
                taxi.drop_all()

        # set dones
        env_done = self.env_done(new_state)
        if self.clear_dead_taxis and not env_done:  # set done for all dead taxis
            dones = {taxi_name: self.__taxi_is_dead(new_state.taxi_by_name[taxi_name]) for taxi_name in self.agents}
        else:  # set done for all taxis when episode is complete
            dones = {taxi_name: env_done for taxi_name in self.agents}

        return new_state, rewards, dones, infos

    def __check_collisions(self, old_state, new_state, infos, rewards):
        # iterate every pair of taxis that can collide
        collidable_taxis = [t for t in new_state.taxis
                            # check if taxi can collide, but also that it is not cleared from the environment
                            if t.can_collide and not (self.clear_dead_taxis and self.__taxi_is_dead(t))]
        for i, taxi1 in enumerate(collidable_taxis):
            for taxi2 in collidable_taxis[i + 1:]:
                # if both collided already, no need to give collision reward again
                if taxi1.collided and taxi2.collided:
                    continue

                collision = False

                # check movement into the same space
                if taxi1.location == taxi2.location:
                    collision = True
                else:
                    # check for position swap
                    old_loc1 = old_state.taxi_by_name[taxi1.name].location
                    old_loc2 = old_state.taxi_by_name[taxi2.name].location
                    if old_loc1 == taxi2.location and old_loc2 == taxi1.location:
                        collision = True

                if collision:
                    for t, other in [(taxi1, taxi2), (taxi2, taxi1)]:
                        if not t.collided:  # check that not already collided in this check
                            t.collided = True
                            self.__add_reward(t.name, infos, rewards, Event.COLLISION)

                        # in any case report collision
                        infos[t.name].setdefault('collided_with', []).append(other.id)

    def __objective_achieved_reward(self, new_state, infos, rewards):
        for taxi in filter(lambda t: t.name in self.agents, new_state.taxis):
            self.__add_reward(taxi.name, infos, rewards, Event.OBJECTIVE)

    def __check_stuck_without_fuel(self, new_state, infos, rewards):
        for taxi in filter(lambda t: t.name in self.agents, new_state.taxis):
            if self.__taxi_stuck_without_fuel(taxi):
                self.__add_reward(taxi.name, infos, rewards, Event.STUCK_WITHOUT_FUEL)

    def __check_out_of_time(self, new_state, infos, rewards):
        for taxi in filter(lambda t: t.name in self.agents, new_state.taxis):
            if taxi.out_of_time:
                self.__add_reward(taxi.name, infos, rewards, Event.OUT_OF_TIME)

    def __check_reduce_fuel(self, taxi, infos, rewards):
        pass

    def __taxi_refuel(self, taxi, infos, rewards):
        # assume refuel would fail and fix if success
        infos[taxi.name]['refuel_success'] = False
        infos[taxi.name]['refueled_units'] = 0

        # get station at taxi location
        station = self.domain_map.fuel_station_at_location(taxi.location)

        if station is None:  # no station
            self.__add_reward(taxi.name, infos, rewards, Event.BAD_REFUEL)
        elif station.fuel_type != taxi.fuel_type:  # wrong station
            self.__add_reward(taxi.name, infos, rewards, Event.BAD_FUEL)
        else:  # success!
            old_fuel = taxi.fuel
            taxi.refuel()
            infos[taxi.name]['refueled_units'] = taxi.fuel - old_fuel
            infos[taxi.name]['refuel_success'] = True
            self.__add_reward(taxi.name, infos, rewards, Event.REFUEL)

    def __taxi_engine_off(self, taxi, infos, rewards):
        if taxi.engine_off:
            self.__add_reward(taxi.name, infos, rewards, Event.USE_ENGINE_WHILE_OFF)
        else:
            taxi.toggle_engine()
            self.__add_reward(taxi.name, infos, rewards, Event.TURN_ENGINE_OFF)

    def __taxi_engine_on(self, taxi, infos, rewards):
        if taxi.engine_on:
            self.__add_reward(taxi.name, infos, rewards, Event.ENGINE_ALREADY_ON)
        elif taxi.empty_tank:
            self.__add_reward(taxi.name, infos, rewards, Event.USE_ENGINE_WHILE_NO_FUEL)
        else:
            taxi.toggle_engine()
            self.__add_reward(taxi.name, infos, rewards, Event.TURN_ENGINE_ON)

    def __taxi_standby(self, taxi, infos, rewards):
        if taxi.engine_on:
            self.__add_reward(taxi.name, infos, rewards, Event.STANDBY_ENGINE_ON)
        else:
            self.__add_reward(taxi.name, infos, rewards, Event.STANDBY_ENGINE_OFF)

    def __move_taxi(self, taxi, move_action, infos, rewards):
        self.__add_reward(taxi.name, infos, rewards, Event.MOVE)
        new_loc = taxi.move(move_action, simulation=True)
        infos[taxi.name]['move_success'] = False  # assume failure case, fix on success
        if not taxi.engine_on:
            self.__add_reward(taxi.name, infos, rewards, Event.USE_ENGINE_WHILE_OFF)
        elif taxi.empty_tank:
            self.__add_reward(taxi.name, infos, rewards, Event.USE_ENGINE_WHILE_NO_FUEL)
        elif self.domain_map.hit_obstacle(taxi.location, new_loc):
            self.__add_reward(taxi.name, infos, rewards, Event.HIT_OBSTACLE)
        else:
            infos[taxi.name]['move_success'] = True
            taxi.move(move_action)

    def __specific_pickup(self, taxi, passengers, action, infos, rewards):
        # assume pickup would fail and fix if success
        infos[taxi.name]['pickup_success'] = False
        if taxi.is_full:
            self.__add_reward(taxi.name, infos, rewards, Event.BAD_PICKUP)
            return  # cannot add passenger. done with action

        # get passenger id for pickup
        passenger_id = int(action[-1])

        for p in passengers:
            if p.id == passenger_id and p.location == taxi.location and not (p.in_taxi or p.arrived):
                taxi.pick_up(p)
                infos[taxi.name]['pickup_success'] = True
                self.__add_reward(taxi.name, infos, rewards, Event.PICKUP)
                return  # added specific passenger. done with action

        if not infos[taxi.name]['pickup_success']:  # pickup did not succeed
            self.__add_reward(taxi.name, infos, rewards, Event.BAD_PICKUP)

    def __generic_pickup(self, taxi, passengers, infos, rewards):
        """
        will pick up any passenger at the taxi's current location.
        if multiple passengers exist at that location, multiple passengers
        """

        # assume pickup would fail and fix if success
        infos[taxi.name]['pickup_success'] = False
        if taxi.is_full:
            self.__add_reward(taxi.name, infos, rewards, Event.BAD_PICKUP)
            return  # cannot add passenger. done with action

        # sort passengers to ensure that we always pick up the passenger with the lowest ID
        passengers = sorted(passengers, key=lambda passenger: passenger.id)

        for p in passengers:
            if p.location == taxi.location and not (p.in_taxi or p.arrived):
                taxi.pick_up(p)
                infos[taxi.name]['pickup_success'] = True
                infos[taxi.name]['picked_up_passenger'] = p.id
                self.__add_reward(taxi.name, infos, rewards, Event.PICKUP)
                break

        if not infos[taxi.name]['pickup_success']:  # pickup did not succeed
            self.__add_reward(taxi.name, infos, rewards, Event.BAD_PICKUP)

    def __specific_dropoff(self, taxi, action, infos, rewards):
        # assume pickup would fail and fix if success
        infos[taxi.name]['dropoff_success'] = False

        # get passenger id for pickup
        passenger_id = int(action[-1])

        for p in taxi.passengers:
            if p.id == passenger_id:
                taxi.drop_off(p)
                infos[taxi.name]['dropoff_success'] = True
                if p.arrived:  # final dropoff
                    self.__add_reward(taxi.name, infos, rewards, Event.FINAL_DROPOFF)
                    infos[taxi.name]['dropped_passenger_at_destination'] = True
                elif self.no_intermediate_dropoff:  # bad intermediate dropoff
                    taxi.pick_up(p)  # undo dropoff
                    infos[taxi.name]['dropoff_success'] = False  # mark as failure
                else:  # intermediate dropoff
                    infos[taxi.name]['dropped_passenger_at_destination'] = False
                    self.__intermediate_dropoff_reward(taxi.name, p, infos, rewards)
                break  # passenger found. no need to continue

        if not infos[taxi.name]['dropoff_success']:  # dropoff did not succeed
            self.__add_reward(taxi.name, infos, rewards, Event.BAD_DROPOFF)

    def __generic_dropoff(self, taxi, infos, rewards):
        # dropoff will succeed as long as there are passengers to drop
        infos[taxi.name]['dropoff_success'] = bool(taxi.passengers)

        # if no passengers on the taxi, mark bad dropoff and continue
        if not infos[taxi.name]['dropoff_success']:
            self.__add_reward(taxi.name, infos, rewards, Event.BAD_DROPOFF)
            return

        # sort passengers to ensure that we always drop off the passenger with the lowest ID
        passengers = sorted(taxi.passengers, key=lambda passenger: passenger.id)

        # assume intermediate dropoff and fix if final dropoff
        infos[taxi.name]['dropped_passenger_at_destination'] = False
        for p in passengers:
            if p.location == p.destination:
                infos[taxi.name]['dropped_passenger_at_destination'] = True
                taxi.drop_off(p)
                self.__add_reward(taxi.name, infos, rewards, Event.FINAL_DROPOFF)
                break

        if not infos[taxi.name]['dropped_passenger_at_destination']:
            # no passengers at their destination
            if self.no_intermediate_dropoff:  # bad dropoff action
                infos[taxi.name]['dropoff_success'] = False  # dropoff failure
                self.__add_reward(taxi.name, infos, rewards, Event.BAD_DROPOFF)
            else:  # intermediate dropoff of the passenger with the lowest ID
                p = passengers[0]
                taxi.drop_off(p)
                self.__intermediate_dropoff_reward(taxi.name, p, infos, rewards)

    def __intermediate_dropoff_reward(self, taxi_name, passenger, infos, rewards):
        # if penalizing by distance, take the negative Manhattan distance from the destination as the reward
        if self.intermediate_dropoff_penalty_by_distance:
            p_row, p_col = passenger.location
            d_row, d_col = passenger.destination
            r = -(abs(p_row - d_row) + abs(p_col - d_col))
        # otherwise, use the reward that appears in the rewards table
        else:
            r = None

        self.__add_reward(taxi_name, infos, rewards, Event.INTERMEDIATE_DROPOFF, r=r)

    def __add_reward(self, taxi_name, infos, rewards, event, r=None):
        rewards[taxi_name] += self.reward_table[taxi_name][event] if r is None else r
        infos[taxi_name]['events'].append(event)

    ######################
    # End Step Functions #
    ######################

    ###########################
    # Done Checking Functions #
    ###########################

    def __taxi_is_done(self, taxi, state=None):
        # taxi is done if one of the following hods true:
        #   1. taxi is dead.
        #   2. objective achieved
        return self.__taxi_is_dead(taxi) or self.__objective_achieved(state)

    def __taxi_can_die(self, taxi_name):
        # a taxi can die if one of the following holds:
        #   1. the taxi can collide
        #   2. the taxi has a finite maximum fuel capacity
        return self.can_collide[taxi_name] or not self.infinite_fuel[taxi_name] or not self.infinite_steps[taxi_name]

    def __taxi_is_dead(self, taxi):
        # taxi is dead before the end of the episode if:
        #   1. taxi has collided.
        #   2. taxi has run out of fuel and is not in a correct fuel station.
        #   3. taxi has surpassed the maximum number of steps it can take
        return taxi.collided or self.__taxi_stuck_without_fuel(taxi) or taxi.out_of_time

    def __objective_achieved(self, state=None):
        if state is None:
            state = self.__state

        # the environment objective has been achieved if one of the following hods true:
        #   1. `pickup_only` is set to `False` and all passengers have arrived at their destinations.
        #   2. `pickup_only` is set to `True` and all passengers are in a taxi.
        return ((not self.pickup_only and all(p.arrived for p in state.passengers)) or  # 1
                (self.pickup_only and all(p.in_taxi for p in state.passengers)))  # 2

    def __taxi_stuck_without_fuel(self, taxi):
        return taxi.empty_tank and not (self.domain_map.at_fuel_station(taxi.location, taxi.fuel_type) and
                                        self.can_refuel_without_fuel[taxi.name])

    ###############################
    # End Done Checking Functions #
    ###############################

    ################################
    # Observation Getter Functions #
    ################################

    def __get_observation_spaces_and_meanings(self):
        obs_spaces = {}
        space_meanings = {}
        for taxi in self.possible_agents:
            obs_type = self.observation_type[taxi]
            if obs_type == ObservationType.SYMBOLIC:
                obs_spaces[taxi], space_meanings[taxi] = self.__get_symbolic_space_and_disambiguation(taxi)
            elif obs_type == ObservationType.IMAGE:
                obs_spaces[taxi], space_meanings[taxi] = self.__get_image_space_and_disambiguation(taxi)
            elif obs_type == ObservationType.MIXED:
                sym_space, sym_meanings = self.__get_symbolic_space_and_disambiguation(taxi)
                img_space, img_meanings = self.__get_image_space_and_disambiguation(taxi)
                obs_spaces[taxi] = spaces.Dict({
                    ObservationType.SYMBOLIC.value: sym_space,
                    ObservationType.IMAGE.value: img_space
                })
                space_meanings[taxi] = {
                    ObservationType.SYMBOLIC.value: sym_meanings,
                    ObservationType.IMAGE.value: img_meanings
                }

            else:
                raise NotImplementedError(f'unsupported observation type {obs_type}')

        return spaces.Dict(obs_spaces), space_meanings

    def __get_symbolic_space_and_disambiguation(self, taxi_name):
        dim_vec = [self.domain_map.map_height, self.domain_map.map_width]
        observations = [SymbolicObservation.LOCATION_ROW.value, SymbolicObservation.LOCATION_COL.value]

        # fuel info
        if not self.infinite_fuel[taxi_name]:
            dim_vec.append(self.max_fuel[taxi_name] + 1)
            observations.append(SymbolicObservation.REMAINING_FUEL.value)

        # engine control info
        if self.has_engine_control[taxi_name]:
            dim_vec.append(2)
            observations.append(SymbolicObservation.ENGINE_ON.value)

        # death indicator
        if self.__taxi_can_die(taxi_name):
            dim_vec.append(2)
            observations.append(SymbolicObservation.IS_DEAD.value)  # death indicator

        # other taxi info
        if self.can_see_other_taxi_info[taxi_name]:
            for name in self.possible_agents:
                if name == taxi_name:
                    continue

                dim_vec.extend([self.domain_map.map_height, self.domain_map.map_width])
                observations.extend([SymbolicObservation.OTHER_LOCATION_ROW.value.format(name=name),
                                     SymbolicObservation.OTHER_LOCATION_COL.value.format(name=name)])

                # fuel info
                if not self.infinite_fuel[name]:
                    dim_vec.append(self.max_fuel[name] + 1)
                    observations.append(SymbolicObservation.OTHER_REMAINING_FUEL.value.format(name=name))

                if self.has_engine_control[name]:
                    dim_vec.append(2)
                    observations.append(SymbolicObservation.OTHER_ENGINE_ON.value.format(name=name))

                # death indicator bit
                if self.__taxi_can_die(name):
                    dim_vec.append(2)
                    observations.append(SymbolicObservation.OTHER_IS_DEAD.value.format(name=name))

        # passengers info
        for i in range(self.num_passengers):
            # passenger location
            dim_vec.extend([self.domain_map.map_height, self.domain_map.map_width]),  # passenger location
            observations.extend([SymbolicObservation.PASSENGER_LOCATION_ROW.value.format(index=i),
                                 SymbolicObservation.PASSENGER_LOCATION_COL.value.format(index=i)])

            # passenger destination and arrival status
            if not self.pickup_only:
                dim_vec.extend([self.domain_map.map_height, self.domain_map.map_width,  # passenger destination
                                2])  # arrived indicator
                observations.extend([SymbolicObservation.PASSENGER_DESTINATION_ROW.value.format(index=i),
                                     SymbolicObservation.PASSENGER_DESTINATION_COL.value.format(index=i),
                                     SymbolicObservation.PASSENGER_ARRIVED.value.format(index=i)])

                # in-taxi indicator bit
                dim_vec.extend([2] * self.num_taxis)  # indicator bit for "in taxi" for each taxi
                observations.extend([SymbolicObservation.PASSENGER_IN_TAX.value.format(p_index=i, t_index=j)
                                     for j in range(self.num_taxis)])
            else:
                # picked up indicator bit
                dim_vec.append(2)  # indicator bit for passenger "picked up"
                observations.append(SymbolicObservation.PASSENGER_PICKED_UP.value.format(index=i))

        return spaces.MultiDiscrete(dim_vec), observations

    def __get_image_space_and_disambiguation(self, taxi_name):
        fov = self.field_of_view[taxi_name]

        # None FOV for taxi, get shape of full map
        if fov is None:
            shape = self.__image_render_helper.image.size[::-1]
        else:
            crop_dims = self.__get_crop_dims(0, 0, fov)
            shape = self.__image_render_helper.get_cur_image_crop(*crop_dims).size[::-1]

        # return RGB space and disambiguation
        return spaces.Box(0, 255, shape + (3,), dtype=np.uint8), f'{"x".join(map(str, shape))} RGB image'

    def __get_action_space_disambiguation(self):
        # general move actions for all taxis
        actions_dict = {taxi: [a.value for a in MOVE_ACTIONS] for taxi in self.possible_agents}

        for taxi in actions_dict:
            # pickup and dropoff action for each passenger
            if self.specify_passenger_pickup[taxi]:
                for i in range(self.num_passengers):
                    actions_dict[taxi].append(f'{Action.PICKUP.value}{i}')
            # only one generic pickup action and one generic dropoff action
            else:
                actions_dict[taxi].append(Action.PICKUP.value)

            if not self.pickup_only:
                if self.specify_passenger_dropoff[taxi]:
                    for i in range(self.num_passengers):
                        actions_dict[taxi].append(f'{Action.DROPOFF.value}{i}')
                else:
                    actions_dict[taxi].append(Action.DROPOFF.value)

            # standby action
            if self.has_standby_action[taxi]:
                actions_dict[taxi].append(Action.STANDBY.value)

            # engine actions
            if self.has_engine_control[taxi]:
                actions_dict[taxi].extend([a.value for a in ENGINE_ACTIONS])

            # add refueling action
            if not self.infinite_fuel[taxi]:
                actions_dict[taxi].append(Action.REFUEL.value)

        # change action lists to dictionaries with index keys for readability
        return {taxi: {i: action for i, action in enumerate(actions_dict[taxi])} for taxi in actions_dict}

    def __observe_all(self):
        return {agent: self.__observe(agent) for agent in self.agents}

    def __observe(self, agent):
        # draw taxis in image in case case needed
        if self.__image_render_helper is not None:
            arr_ansi = self.__render_map(as_numpy=True)
            self.__image_render_helper.draw_taxis_and_passengers(arr_ansi)

        obs_type = self.observation_type[agent]
        if obs_type == ObservationType.SYMBOLIC:
            return self.__symbolic_observe(agent)
        elif obs_type == ObservationType.IMAGE:
            return self.__image_observe(agent)
        elif obs_type == ObservationType.MIXED:
            return {
                ObservationType.SYMBOLIC.value: self.__symbolic_observe(agent),
                ObservationType.IMAGE.value: self.__image_observe(agent)
            }
        else:
            raise NotImplementedError(f'unsupported observation type {obs_type}')

    def __symbolic_observe(self, taxi_name):
        cur_taxi = self.__state.taxi_by_name[taxi_name]
        obs = list(cur_taxi.location)

        # fuel info
        if not self.infinite_fuel[taxi_name]:
            obs.append(cur_taxi.fuel)

        # engine control
        if self.has_engine_control[taxi_name]:
            obs.append(cur_taxi.engine_on)

        # death indicator
        if self.__taxi_can_die(taxi_name):
            obs.append(self.__taxi_is_dead(cur_taxi))

        # other taxi info
        if self.can_see_other_taxi_info[taxi_name]:
            for i, other_taxi in enumerate(self.__state.taxis):
                if taxi_name == other_taxi.name:
                    continue

                obs.extend(list(other_taxi.location))

                # fuel info
                if not self.infinite_fuel[other_taxi.name]:
                    obs.append(other_taxi.fuel)

                # engine control
                if self.has_engine_control[other_taxi.name]:
                    obs.append(other_taxi.engine_on)

                # death indicator
                if self.__taxi_can_die(other_taxi.name):
                    obs.append(self.__taxi_is_dead(other_taxi))

        # passengers info
        for i, passenger in enumerate(self.__state.passengers):
            # passenger location
            obs.extend(list(passenger.location))

            # passenger destination and arrival status
            if not self.pickup_only:
                obs.extend(list(passenger.destination) + [passenger.arrived])

            # in-taxi indicator bit
            obs.extend([passenger.carrying_taxi == j for j in range(self.num_taxis)])

        return np.array(obs)

    def __image_observe(self, taxi_name):
        taxi_fov = self.field_of_view[taxi_name]

        if taxi_fov is None:  # use entire map
            pil_img = self.__image_render_helper.cur_img

        # given FOV, crop out FOV square window
        else:
            taxi_loc = self.__state.taxi_by_name[taxi_name].location
            taxi_map_row, taxi_map_col = self.domain_map.location_to_domain_map_idx(*taxi_loc)
            crop_dims = self.__get_crop_dims(taxi_map_row, taxi_map_col, taxi_fov)
            pil_img = self.__image_render_helper.get_cur_image_crop(*crop_dims)

        return np.array(pil_img)

    @staticmethod
    def __get_crop_dims(center_row, center_col, fov):
        fov_h = fov
        fov_w = fov * 2

        # find crop indices
        bottom_crop = center_row + fov_h + 1
        top_crop = center_row - fov_h
        right_crop = center_col + fov_w + 3
        left_crop = center_col - fov_w - 1

        return bottom_crop, top_crop, right_crop, left_crop

    ####################################
    # End Observation Getter Functions #
    ####################################

    #######################
    # Rendering Functions #
    #######################

    def __render_map(self, with_colors=True, as_numpy=False):
        # take domain map as is
        out_map = self.domain_map.domain_map

        # mark passengers and destinations
        for passenger in self.__state.passengers:
            if not passenger.arrived:  # only show passengers that are not in their destination
                if not passenger.in_taxi:  # only show passenger when not in taxi
                    map_row, map_col = self.domain_map.location_to_domain_map_idx(*passenger.location)
                    out_map[map_row, map_col] = self.__cond_colorize(PASSENGER_RENDERING_CHAR, passenger.color,
                                                                     highlight=False, with_colors=with_colors)
                if not self.pickup_only:  # only show destinations if "pickup only" is disabled
                    map_row, map_col = self.domain_map.location_to_domain_map_idx(*passenger.destination)
                    out_map[map_row, map_col] = self.__cond_colorize(DESTINATION_RENDERING_CHAR, passenger.color,
                                                                     highlight=False, with_colors=with_colors)

        # highlight taxis on the map
        for taxi in self.__state.taxis:
            # skip dead taxi
            if self.__taxi_is_dead(taxi) and self.clear_dead_taxis:
                continue

            map_row, map_col = self.domain_map.location_to_domain_map_idx(*taxi.location)

            # check existence of same colored passenger/destination and change to original black
            if out_map[map_row, map_col] == colorize(PASSENGER_RENDERING_CHAR, taxi.color, highlight=False):
                out_map[map_row, map_col] = PASSENGER_RENDERING_CHAR
            elif out_map[map_row, map_col] == colorize(DESTINATION_RENDERING_CHAR, taxi.color, highlight=False):
                out_map[map_row, map_col] = DESTINATION_RENDERING_CHAR

            # highlight existing test
            out_map[map_row, map_col] = colorize(out_map[map_row, map_col], taxi.color, highlight=True)

        if as_numpy:
            return out_map
        else:
            # return as a string
            return '\n'.join([''.join(row) for row in out_map])

    @staticmethod
    def __cond_colorize(string, color, bold=False, highlight=False, with_colors=True):
        if with_colors:
            return colorize(string, color, bold, highlight)
        else:
            return string

    def __render_status(self):
        status_str = ''

        # output taxi data
        for taxi in self.__state.taxis:
            status_str += f'{taxi}, {"DEAD" if self.__taxi_is_dead(taxi) else "ALIVE"}\n'

        # output passengers data
        for passenger in self.__state.passengers:
            status_str += f'{passenger}\n'

        # output "env done"
        status_str += f'Env done: {self.env_done()}\n'

        return status_str

    ###########################
    # End Rendering Functions #
    ###########################
