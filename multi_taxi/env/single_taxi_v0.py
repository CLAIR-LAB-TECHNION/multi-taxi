from . import multi_taxi_v0 as __multi_taxi_v0
from ..wrappers import SingleTaxiWrapper as __SingleTaxiWrapper


def gym_env(*args, **kwargs):
    multi_taxi_env = __multi_taxi_v0.parallel_env(*args, **kwargs)

    # assert single taxi if "num_taxis" is not given (in case of defaults change)
    # if "num_taxis" was already given, fail on wrapper assertion.
    if not args and 'num_taxis' not in kwargs:
        kwargs['num_taxis'] = 1

    single_taxi_env = __SingleTaxiWrapper(multi_taxi_env)
    single_taxi_env.unwrapped.metadata['name'] = 'single_taxi_v0'
    return single_taxi_env
