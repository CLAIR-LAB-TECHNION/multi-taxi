from .env_creator import EnvCreator

from multi_taxi import multi_taxi_v0, maps, FuelType
from multi_taxi.env.reward_tables import TAXI_ENVIRONMENT_REWARDS
from multi_taxi.utils.types import Event

class MultiTaxiCreator(EnvCreator):

    ENV_NAME = "multi_taxi_env"

    @staticmethod
    def get_env_name():
        return "multi_taxi_env"

    @staticmethod
    def create_env():
        # Setup a custom reward table base on the default one
        custom_reward_table = TAXI_ENVIRONMENT_REWARDS
        custom_reward_table[Event.BAD_DROPOFF] = -10
        custom_reward_table[Event.BAD_PICKUP] = -10
        custom_reward_table[Event.FINAL_DROPOFF] = 1000
        custom_reward_table[Event.PICKUP] = 10
        custom_reward_table[Event.OUT_OF_TIME] = -200
        custom_reward_table[Event.STEP] = -1

        # using the PettingZoo parallel API here
        return multi_taxi_v0.parallel_env(
            num_taxis=2,                       # there are 2 active taxis (agents) in the environment
            num_passengers=3,                  # there are 3 passengers in the environment
            max_steps=1000,
            reward_table=custom_reward_table,
            intermediate_dropoff_reward_by_distance=True,
            max_capacity=[1, 2],               # taxi_0 can carry 1 passenger, taxi_1 can carry 2
            max_fuel=[None, None],               # taxi_0 has a 30 step fuel limit, taxi1 has infinite fuel
            fuel_type=FuelType.GAS,            # taxis can only refuel at gas stations, marked "G" (only affects taxi_0)
            has_standby_action=True,           # all taxis can perform the standby action
            has_engine_control=[False, False],  # taxi_0 has engine control actions, taxi_1 does not
            domain_map=maps.SMALL_NO_OBS_MAP,   # the environment map is the pre-defined HOURGLASS map
            render_mode='human'  # MUST SPECIFY RENDER MODE TO ENABLE RENDERING
        )