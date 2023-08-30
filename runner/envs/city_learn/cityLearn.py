from runner.envs.env_creator import EnvCreator
import subprocess
subprocess.run(["pip", "install", "git+https://github.com/CLAIR-LAB-TECHNION/CLAIR_grid"], check=True)

import itertools
from clair_grid.coordination.sg_coordinator import SGDecentralizedCoordinator, SGCentralizedCoordinator, SGDecentralizedWithComCoordinator
from clair_grid.environment.citylearn_wrapper import CityLearnWrapper
from clair_grid.agents.agent import Agent

from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import RewardFunction
from typing import List
from clair_grid.agents.sg_random_agent import ESRandomAgent, CentralESRandomAgent
from citylearn.reward_function import SolarPenaltyReward, RewardFunction, IndependentSACReward
from citylearn.energy_model import ZERO_DIVISION_CAPACITY


class CityLearnCreator(EnvCreator):
    ENV_NAME = "city_learn_env"

    @staticmethod
    def get_env_name():
        return "city_learn_env"

    @staticmethod
    def create_env():
        building_indices = None
        # schema_path = '~/data/schema.json'

        # Initialize the environment
        dataset_name = 'citylearn_challenge_2022_phase_1'
        # env = CityLearnEnv(schema=schema_path)
        env = CityLearnEnv(schema=dataset_name)
        env.reward_function = SolarPenaltyReward(env)
        env_wrapper = CityLearnWrapper(env)

        # create the agents
        num_of_agents = len(env.buildings)

        # # initialize the coodinator
        # b_random_order = False
        # coordinator = SGDecentralizedCoordinator(env_wrapper, agents, agent_ids, b_random_order)
        #
        # # run training
        # coordinator.run(1000, b_log=True, b_train=True, b_evaluate=False)

        # If building_indices are provided, select only the indicated buildings.
        if building_indices:
            env.buildings = [env.buildings[i] for i in building_indices]

        num_of_agents = len(env.buildings)

        return env
