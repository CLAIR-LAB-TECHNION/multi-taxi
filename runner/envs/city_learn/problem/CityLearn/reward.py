from typing import List

import numpy as np
from citylearn.reward_function import RewardFunction


class CustomReward(RewardFunction):
    def __init__(self, env):
        super().__init__(env=env)

    # def calculate(self) -> List[float]:
    #     """CityLearn Challenge reward calculation.
    #     This function is called internally in the environment's :meth:`citylearn.CityLearnEnv.step` function.
    #     CityLearn Challenge user reward calculation.
    #
    #     electricity_consumption: List[float]
    #         List of each building's/total district electricity consumption in [kWh].
    #     carbon_emission: List[float]
    #         List of each building's/total district carbon emissions in [kg_co2].
    #     electricity_price: List[float]
    #         List of each building's/total district electricity price in [$].
    #     agent_ids: List[int]
    #         List of agent IDs matching the ordering in `electricity_consumption`, `carbon_emission` and `electricity_price`.
    #
    #     Returns
    #     -------
    #     rewards: List[float]
    #         Agent(s) reward(s) where the length of returned list is either = 1 (central agent controlling all buildings)
    #         or = number of buildings (independent agent for each building).
    #     """
    #
    #     carbon_emission_index = 19
    #     electricity_consumption_index = 23
    #     electricity_pricing_index = 24
    #     observation = self.env.observations
    #
    #     electricity_consumption = np.array([o[electricity_consumption_index] for o in observation])
    #     carbon_emission = np.array([o[carbon_emission_index] * o[electricity_consumption_index] for o in observation])
    #     electricity_price = np.array([o[electricity_pricing_index] * o[electricity_consumption_index] for o in observation])
    #
    #     reward = -(electricity_consumption + carbon_emission + electricity_price)
    #     return reward

    def calculate(self) -> List[float]:
        if self.env.central_agent:
            reward = [self.env.net_electricity_consumption_emission[-1]]
        else:
            reward = [b.net_electricity_consumption_emission[-1] for b in self.env.buildings]

        return reward

    # def generic_cost_calculate(self, weights, costs):
    #     assert weights.length() == costs.lenght(), "Make sure weights vector is same size as chosen costs vector"
    #     sum_cost = 0
    #     sum_reward = 0
    #     for i, c in enumerate(costs):
    #         sum_cost = sum_cost + weights[i] * c
    #         sum_reward = sum_reward - weights[i] * c
    #     return sum_cost, sum_reward
