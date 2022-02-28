from MultiTaxiLib.taxi_environment import TaxiEnv
from MultiTaxiLib.taxi_utils.rendering_utils import map2rgb
import random

import ray
from ray import tune
from ray.rllib.examples.models.shared_weights_model import TorchSharedWeightsModel

from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec


def test__get_observation_space_list():
    taxi_env = TaxiEnv()
    obs_space_list = taxi_env._get_observation_space_list()
    assert obs_space_list == [7, 12, 1, 7, 12, 7, 12, 4]


def test_reset():
    obs_image = TaxiEnv(observation_type='image').reset()
    obs_symbolic = TaxiEnv().reset()
    assert obs_image['taxi_1'].shape == (5, 9, 3)
    assert obs_symbolic['taxi_1'].shape == (8,)


def test_map2rgb_fuel_station_bag():
    taxi_env = TaxiEnv(observation_type='image')
    orig_map = taxi_env.desc
    rgb_arr = map2rgb(taxi_env.state, orig_map)
    for station_coord in taxi_env.fuel_stations:
        assert all(rgb_arr[station_coord[0] + 1, station_coord[1] * 2 + 1] != rgb_arr[0, 0])


def test_able_to_run_rllib_agents():
    ray.init(ignore_reinit_error=True, local_mode=True)

    # Register the models to use.
    mod1 = mod2 = TorchSharedWeightsModel
    ModelCatalog.register_custom_model("model1", mod1)
    ModelCatalog.register_custom_model("model2", mod2)

    num_policies = 1
    num_agents = 1

    # Each policy can have a different configuration (including custom model).
    def gen_policy(i):
        config = {
            "model": {
                "custom_model": ["model1", "model2"][i % 2],
            },
            "gamma": random.choice([0.95, 0.99]),
        }
        return PolicySpec(config=config)

    # Setup PPO with an ensemble of `num_policies` different policies.
    policies = {"taxi_{}".format(i + 1): gen_policy(i) for i in range(num_policies)}
    policy_ids = list(policies.keys())

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return agent_id

    config = {
        "env": TaxiEnv,
        "env_config": {
            "taxis_number": num_agents,
            "can_see_others": True
        },
        "num_gpus": 0,
        "horizon": 50,
        "simple_optimizer": True,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
        },
        "framework": "torch",
    }
    stop = {
        "episode_reward_mean": 50,
        "timesteps_total": 1000,
        "training_iteration": 10,
    }

    results = tune.run("DQN", stop=stop, config=config, verbose=1)
    # check_learning_achieved(results, 10000)
    print(results.dataframe())
    ray.shutdown()
