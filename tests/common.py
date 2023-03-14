import json
from pathlib import Path

from multi_taxi import maps

ENV_CFG_FILE = Path(__file__).parent / 'test_env_configs.json'


def get_test_env_cfgs():
    with open(ENV_CFG_FILE, 'r') as f:
        env_cfgs = json.load(f)

    # handle special input
    for cfg in env_cfgs.values():

        # domain map given by name of predefined map
        if 'domain_map' in cfg and isinstance(cfg['domain_map'], str):
            cfg['domain_map'] = getattr(maps, cfg['domain_map'])

    return env_cfgs


test_env_cfgs = get_test_env_cfgs()
single_agent_cfgs = {k: cfg for k, cfg in test_env_cfgs.items() if 'num_taxis' not in cfg or cfg['num_taxis'] == 1}
