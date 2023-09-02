import numpy as np
from gymnasium.vector.utils import spaces
from pettingzoo import ParallelEnv

from env_creator import EnvCreator

# pip install git+https://github.com/CLAIR-LAB-TECHNION/CLAIR_grid
# from clair_grid.environment.citylearn_wrapper import CityLearnWrapper

from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import SolarPenaltyReward


class CityLearnWrapper(ParallelEnv):
    def __init__(self, building_indices=None):
        # Path to the environment schema
        schema_path = 'envs/city_learn/data/schema.json'
        max_num_buildings = 17
        # dataset_name = 'citylearn_challenge_2022_phase_1'
        env = CityLearnEnv(schema=schema_path)
        env.reward_function = SolarPenaltyReward(env)

        if building_indices:
            env.buildings = [env.buildings[i] for i in building_indices]

        self.env = env

        self.observation_space = lambda building: spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)
        # self.action_space = lambda building: gym.spaces.Discrete(n=3, start=-1)

    def reset(self):
        return self.env.reset()

    # @property
    # def observation_space(self):
    #     def observation_space_wrapper(building):
    #         return self.env.observation_space[0]  # [int(building.name.split('_')[1])-1]
    #     return observation_space_wrapper

    @property
    def agents(self):
        return self.env.buildings

    # @property
    def action_space(self, building):
        return self.env.action_space[int(building.name.split('_')[1])-1]
        # return action_space_wrapper

    def step(self, action):
        return self.env.step(action)

    def get_agent_step_data(self, step_data, agent_id):
        # agent id's start at 1 and are stored according to order in a list
        [observations, rewards, done, info] = step_data
        agent_step_data = [observations[agent_id - 1], rewards[agent_id - 1], done]
        return agent_step_data

    def transform_action_dict_to_env_format(self, actions:dict):
        action_list = [None] * len(actions)
        for agent_id in actions.keys():
            action_list[agent_id - 1] = [actions[agent_id]]

        return action_list

    def transform_action_env_format_to_dict(self, actions) -> dict:
        action_dict = {}
        for agent_index in range(0,len(actions)):
            agent_id = agent_index+1
            action_dict[agent_id] = actions[agent_index]

        return action_dict

    def get_kpi_value(self, kpi_index, agent_index = None):
        pass


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
        # dataset_name = 'citylearn_challenge_2022_phase_1'
        # env = CityLearnEnv(schema=schema_path)
        # env = CityLearnEnv(schema=dataset_name)
        # env.reward_function = SolarPenaltyReward(env)
        # env = CityLearnWrapper(env)

        # create the agents
        # num_of_agents = len(env.buildings)

        # # initialize the coodinator
        # b_random_order = False
        # coordinator = SGDecentralizedCoordinator(env_wrapper, agents, agent_ids, b_random_order)
        #
        # # run training
        # coordinator.run(1000, b_log=True, b_train=True, b_evaluate=False)

        # If building_indices are provided, select only the indicated buildings.

        return CityLearnWrapper(building_indices)
