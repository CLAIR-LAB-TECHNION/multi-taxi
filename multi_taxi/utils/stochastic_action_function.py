from collections import defaultdict, OrderedDict
from itertools import product
from typing import Dict

import numpy as np


class JointStochasticActionFunction:
    def __init__(self, probs_dict_dict):
        self.saf_dict = defaultdict(lambda: StochasticActionFunction({}))
        for agent, probs_dict in probs_dict_dict.items():
            self.saf_dict[agent] = StochasticActionFunction(probs_dict)

    def __call__(self, joint_action, rng=np.random):
        return {agent: self.saf_dict[agent](action, rng) for agent, action in joint_action.items()}

    def __getitem__(self, item):
        return self.saf_dict[item]

    def iterate_possible_actions(self, joint_action):
        # make ordered dict for agent name preservation when yielding a dictionary
        joint_action = OrderedDict(joint_action.items())

        # list of lists of (action, probability) pairs
        possible_actions = [list(self.saf_dict[agent].get_actions_probs(action).items())
                            if action in self.saf_dict[agent].probs_dict
                            else [(action, 1.0)]  # action not stochastic
                            for agent, action in joint_action.items()]

        # iterate all possible action combinations
        for actions_and_probs_list in product(*possible_actions):
            # separate actions and their probabilities
            actions_list, probs = zip(*actions_and_probs_list)

            # calculate probability for actions list
            prob = np.product(probs)

            # yield actions dict and actions dict probability
            yield {agent: action for agent, action in zip(joint_action.keys(), actions_list)}, prob


class StochasticActionFunction:
    """
    A function class that uses a pre-defined action probability distribution to determine action outcomes.
    """

    def __init__(self, probs_dict: Dict[str, Dict[str, float]]):
        """
        Initialize the function with a discrete probability distribution using a dictionary of dictionaries.
        Args:
            probs_dict: a dictionary of dictionaries defining a conditional probability distribution like so:
                            probs_dict[desired_action][output_aciton] = p(output_aciton|desired_action)
                        The dictionary can have missing values. Where the distribution is not defined, we assume a
                        deterministic action, i.e.
                            p(desired_action|desired_action) = 1
                        If a distribution is provided for some action, all probabilities must sum to 1.
                        Example:
                            >>> action_dist = {
                            >>>     'north': {'north': 0.7, 'east': 0.15, 'west': 0.15},
                            >>>     'south': {'south': 0.7, 'east': 0.15, 'west': 0.15}
                            >>> }
                            >>> f = StochasticActionFunction(action_dist)
                        In the above example, choosing to perform 'north' or 'south' actions have a 15% chance to
                        perform the 'east' action and a 15% chance to perform 'west' action instead of the given
                        north/south action. All other actions are performed as as expected.
        """
        self.probs_dict = probs_dict

        # iterate all actions to check the conditional distributions' validity.
        for action in self.probs_dict:

            # list conditional probabilities for `action`
            action_probs = list(probs_dict[action].values())

            # check that all probabilities are between 0 and 1
            if not all(0 <= p <= 1 for p in action_probs):
                raise ValueError(f'Bad probability distribution for action {action}: '
                                 f'values must be in [0, 1], but got {probs_dict[action]}')

            # check that the sum is exactly 1
            probs_sum = np.sum(action_probs)
            if probs_sum != 1:
                raise ValueError(f'Bad probability distribution for action {action}: '
                                 f'sum of probabilities is greater than 1: sum({probs_dict[action]}) = {probs_sum}')

    def __call__(self, chosen_action, rng=np.random):
        """
        chooses an action based on the instance's stochastic distribution (the `probs_dict` property).
        Args:
            chosen_action: the action key
        """
        possible_actions_dict = self.get_actions_probs(chosen_action)

        # create a sequence of possible actions and corresponding probabilities.
        possible_actions, action_probs = zip(*possible_actions_dict.items())

        # sample one of the possible actions, weighted by their given probabilities.
        return rng.choice(possible_actions, p=action_probs)

    def get_actions_probs(self, chosen_action):
        if chosen_action not in self.probs_dict or not self.probs_dict[chosen_action]:
            return {chosen_action: 1.0}
        else:
            return self.probs_dict[chosen_action]
