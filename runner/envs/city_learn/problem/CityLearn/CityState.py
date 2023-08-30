import numpy as np
from copy import deepcopy


class CityState:
    def __init__(self, problem, env, parent, action, observation, done, rewards, discretisize_states):
        """
        Initialize a new city state.

        Args:
            problem: The SuperProblem instance.
            env: The CityLearn environment.
            done: A boolean indicating whether the simulation is complete.
            rewards: A list of accumulated rewards.
            discretisize_states: A boolean indicating whether to discretize states.
        """
        self.problem = problem
        self.env = env
        self.parent = parent
        self.action = action
        self.observation = observation
        self.rewards = rewards
        self.done = done
        self.index = problem.update_data(self)
        self.discretisize_states = discretisize_states

    def depth(self):
        return len(self.path())

    def successor(self, action):
        """
        Generate the successor state given an action.

        Args:
            action: The action to be performed.

        Returns:
            A new city state that results from performing the action.
        """
        env = deepcopy(self.env)
        observation, reward, done, info = env.step(action)

        if self.discretisize_states:
            for building in env.buildings:
                features = [f for f in ["cooling_storage_soc", "heating_storage_soc",
                            "dhw_storage_soc", "electrical_storage_soc"] if f in building.active_observations]
                for f in features:
                    f = f"_Building__{f[:-4]}"
                    values = np.linspace(0, building.__dict__[f].capacity, self.problem.soc_num_values)
                    building.__dict__[f].soc[-1] = min(values, key=lambda v: abs(v - building.__dict__[f].soc[-1]))

        return CityState(self.problem, env, self, action, observation, done, self.rewards + [sum(reward)], self.discretisize_states)

    def path(self):
        path = [(self.action, self)]
        while path[0][1].parent:
            path = [(path[0][1].parent.action, path[0][1].parent)] + path
        return path

    def get_applicable_actions(self):
        """
        Retrieve the applicable actions for the current state.

        Returns:
            A list of applicable actions. (Currently not implemented.)
        """
        # return self.discretization.get()
        raise NotImplementedError()

    def get_transition_path_string(self):
        """
        Retrieve the transition path as a string.

        Returns:
            An empty string. (Currently not implemented.)
        """
        return self.path()

    def is_done(self):
        """
        Check if the current state is a terminal state or not.

        Returns:
            A boolean indicating whether the maximum number of states has been discovered
            or if the simulation has been completed.
        """
        return self.done or \
            self.problem.discovered_states >= self.problem.max_states or \
            self.depth() >= self.problem.min_steps_to_finish

    def result(self):
        """
        Compute the average of the accumulated rewards.

        Returns:
            The average of the accumulated rewards.
        """
        return np.mean(self.rewards) if self.rewards else 0

    def __str__(self):
        return f"[{self.index}|{self.depth()}|{self.result():.4f}]"

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        return self.result() < other.result()
