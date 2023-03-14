import pytest
from pettingzoo.test import api_test, parallel_api_test

from multi_taxi import multi_taxi_v0
from .common import test_env_cfgs

NUM_CYCLES = 1000
NUM_SEEDS = 5
INITIAL_SEED = 42


@pytest.mark.parametrize(['env_name', 'env_cfg'], test_env_cfgs.items())
def test_aec_api(env_name, env_cfg):
    __multi_seed_api_test(multi_taxi_v0.env, env_name, env_cfg, api_test)


@pytest.mark.parametrize(['env_name', 'env_cfg'], test_env_cfgs.items())
def test_parallel_api(env_name, env_cfg):
    __multi_seed_api_test(multi_taxi_v0.parallel_env, env_name, env_cfg, parallel_api_test)


def __multi_seed_api_test(env_init, name, cfg, pz_test):
    seed = hash(name)
    for _ in range(NUM_SEEDS):
        seed = abs(hash(str(seed)))
        pz_test(env_init(**cfg, initial_seed=seed), num_cycles=NUM_CYCLES)
