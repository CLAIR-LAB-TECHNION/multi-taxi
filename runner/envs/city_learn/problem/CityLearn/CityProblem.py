from copy import deepcopy

import numpy as np
from citylearn.citylearn import CityLearnEnv
from matplotlib import pyplot as plt

from problem.CityLearn.CityState import CityState
from problem.CityLearn.kpi__ import plot_simulation_summary
from search_algorithms.utils import State, Node
from problem.SuperProblem import SuperProblem


class CityProblem(SuperProblem):
    # Path to the environment schema
    schema_path = 'data/schema.json'
    max_num_buildings = 17

    # Variables controlling problem termination conditions
    min_steps_to_finish = 5
    max_states = 50000

    # Variables controlling state discretization
    discretisize_states = False
    feature_num_values = 4
    soc_num_values = 10

    def __init__(self, building_indices):
        """
        Initialize a new city problem instance.

        Args:
            building_indices: A list of indices for the buildings to consider in the problem.
        """
        # Initialize the environment
        self.env = CityLearnEnv(schema=self.schema_path)
        self.discretization = None

        # Select only the buildings indicated by the indices
        if not building_indices:
            building_indices = list(range(self.max_num_buildings))
        self.env.buildings = [self.env.buildings[i] for i in building_indices]
        self.building_indices = building_indices

        # Initialize problem metrics
        self.discovered_states = 0
        self.best_result = []
        self.deepest_depth = 0

        # Initialize state discretization if enabled
        if self.discretisize_states:
            self.init_states_discretization()

        # Initialize the root state and SuperProblem [1=expended state|1=depth|0.000=reward]
        initial_state = CityState(self, self.env, None, None, None, False, [], self.discretisize_states)

        super().__init__(initial_state=initial_state, constraints=[])

    def init_states_discretization(self):
        """
        Initialize the discretization of the states space.
        """
        # Get the observations and their boundaries
        names = self.env.observation_names[0]
        num_features = len(names)
        low, high = self.env.observation_space[0].low, self.env.observation_space[0].high

        # Define discrete features indices
        discrete_features_indices = {'month': 12, 'day_type': 7, 'hour': 24}
        features_values = {}

        # Define epsilon values for staying within bounds
        epsilon = 1e-5

        # Create a mapping from feature names to discretized values
        for i in range(num_features):
            if names[i] in discrete_features_indices:
                num_values = discrete_features_indices[names[i]]
            else:
                num_values = self.feature_num_values

            features_values[names[i]] = [low[i] + epsilon, high[i] - epsilon,
                                         np.linspace(low[i], high[i], num_values)]

        # TODO: checks features in https://www.citylearn.net/overview/schema.html
        prefix = "_Building__"
        for building in self.env.buildings:
            features = [x[len(prefix):] for x in vars(building) if x.startswith(prefix)]
            # features = ["energy_simulation", "weather", "pricing", "carbon_intensity"]
            for f in features:
                f = building.__dict__[f"{prefix}{f}"]
                if "__dict__" not in dir(f):
                    continue
                fields = [x for x in vars(f).keys() if x in building.active_observations]
                for x in fields:
                    arr = f.__dict__[x]
                    for i in range(arr.size):
                        arr[i] = min(features_values[x][2], key=lambda v: abs(v - arr[i]))

    def update_data(self, state):
        """
        Update data related to the problem's state and keep track of some metrics.

        Args:
            state: The current state.

        Returns:
            The total number of discovered states.
        """
        self.discovered_states += 1

        if state.depth() > self.deepest_depth:
            self.deepest_depth = state.depth()

        if state.depth() >= self.min_steps_to_finish and \
                (self.best_result == [] or state.result() > self.best_result[-1][2].result()):
            self.best_result.append((self.discovered_states, self.deepest_depth, state))

        return self.discovered_states

    def get_applicable_actions_at_state(self, state):
        """
        Retrieve the applicable actions for the given state.

        Args:
            state: The state to retrieve actions for.

        Returns:
            A list of applicable actions. (Currently not implemented.)
        """
        # return state.get_key().get_applicable_actions()[:]
        raise NotImplementedError()

    def get_applicable_actions_at_node(self, node):
        return self.get_applicable_actions_at_state(node.state)[:]

    def get_successors(self, action=None, node=None):
        """
        Get successors of the current node.

        Args:
            action: The action to perform
            node: The current node.

        Returns:
            A list of successor nodes.
        """
        assert node is not None

        actions = [action] if action else node.discretization.update()

        all_successors = []

        for action in actions:
            successor = node.state.get_key().successor(action)
            successor_state = State(successor, successor.is_done())
            cost = self.get_action_cost(action, successor_state)
            next_node = Node(successor_state, node, action, node.path_cost + cost,
                             deepcopy(node.discretization))
            all_successors.append(next_node)

        return all_successors

    def get_action_cost(self, action, state):
        """
        Calculate the rewards of the action.

        Args:
            action: The action to be performed.
            state: The state to which the action leads.

        Returns:
            The absolute value of the last reward in the state.
        """
        # TODO find better calculation?
        return -state.get_key().rewards[-1]

    def is_goal_state(self, state):
        """
        Check if a state is a goal state.

        Args:
            state: The state to check.

        Returns:
            A boolean indicating whether the state is a goal state or not.
        """
        return state.get_key().is_done()

    def apply_action(self, action):
        """
        Apply an action to the environment. Currently not implemented.

        Args:
            action: The action to apply.
        """
        raise NotImplementedError()

    def reset_env(self):
        """
        Reset the CityLearn environment.

        Returns:
            The initial state of the environment after reset.
        """
        return self.env.reset()

    def summary(self):
        """
        Get the result of the current problem.

        Returns:
            A tuple consisting of the total number of discovered states, the deepest depth achieved,
            and the best result found.
        """
        best_state = self.best_result[-1][2]
        path = best_state.path()
        summary = f"Number of discovered states: {self.discovered_states}\n" \
                  f"Max depth: {self.deepest_depth}\n" \
                  f"Best result: {best_state.result()}\n" \
                  f"Path: {path}\n" \
                  f"Environment evaluation:\n{best_state.env.evaluate()}"

        electrical_storage_soc_index = self.env.observation_names[0].index('electrical_storage_soc')
        electrical_storage_soc = [[x[1].observation[i][electrical_storage_soc_index] for x in path if x[1].observation]
                                  for i in range(len(self.building_indices))]
        for i, b in enumerate(self.building_indices):
            plt.plot(list(range(len(electrical_storage_soc[i]))), electrical_storage_soc[i], label=f"Building {b}")
        plt.xlabel("time")
        plt.ylabel("electrical_storage_soc")
        plt.title("electrical_storage_soc")
        plt.legend()
        plt.show()

        plot_simulation_summary(best_state.env)

        return summary

    def evaluation_criteria(self):
        return CityProblemEvaluationCriteria()


class CityProblemEvaluationCriteria:
    def is_better_or_equal(self, cur_value, cur_node, best_value, best_node, problem):
        if cur_node.state.get_key().depth() >= problem.min_steps_to_finish > best_node.state.get_key().depth():
            return True
        return abs(cur_node.state.get_key().result()) < abs(best_node.state.get_key().result())
