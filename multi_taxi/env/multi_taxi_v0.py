from pettingzoo.utils import wrappers as __pz_wrappers, parallel_to_aec as __parallel_to_aec

from .MultiTaxiEnv import MultiTaxiEnv

from .. import wrappers as __custom_wrappers


def env(*args, **kwargs):
    aec_env = raw_env(*args, **kwargs)
    aec_env = __pz_wrappers.AssertOutOfBoundsWrapper(aec_env)
    aec_env = __pz_wrappers.OrderEnforcingWrapper(aec_env)
    return aec_env


def raw_env(*args, **kwargs):
    p_env = MultiTaxiEnv(*args, **kwargs)
    return __parallel_to_aec(p_env)


def parallel_env(*args, **kwargs):
    p_env = MultiTaxiEnv(*args, **kwargs)
    p_env = __custom_wrappers.AssertOutOfBoundsParallelWrapper(p_env)
    p_env = __custom_wrappers.OrderEnforcingParallelWrapper(p_env)
    return p_env
