
import os, sys

dir2 = os.path.abspath('/home/ofir/PycharmProjects/DFC/Domains')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)

from MultiTaxiLib.taxi_environment import TaxiEnv
import matplotlib.pyplot as plt
from MultiTaxiLib.taxi_utils import rendering_utils
from contextlib import contextmanager, redirect_stdout
from io import StringIO
import streamlit as st
import numpy as np


@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret

        stdout.write = new_write
        yield


def environment_snapshot(environment_instance: TaxiEnv) -> None:
    output = st.empty()

    print("Grid world representation:")
    rendering_utils.render(environment_instance.desc.tolist(), environment_instance.state,
                           environment_instance.num_taxis, environment_instance.collided,
                           environment_instance.last_action, environment_instance.action_index_dictionary,
                           environment_instance.dones)

    st.write("Image world representation:")
    img = rendering_utils.map2rgb(environment_instance.state, environment_instance.desc.astype(str))
    fig, ax = plt.subplots()
    ax.imshow(img)
    st.pyplot(fig)


def get_action_dictionary_from_list(action_index_dict: dict, agent_names: list, action_list: list) -> dict:
    action_dict = {}
    for i, name in enumerate(agent_names):
        action_dict[name] = action_index_dict[action_list[i]]

    return action_dict


def observation_snapshot(observations: dict) -> None:
    num_of_observations = len(list(observations.keys()))
    fig, axs = plt.subplots(1, num_of_observations)
    for i, name in enumerate(list(observations.keys())):
        if num_of_observations > 1:
            axs[i].imshow(observations[name][0].astype(np.uint8))
            axs[i].title.set_text(f"{name}")
        else:
            axs.imshow(observations[name][0].astype(np.uint8))
            axs.title.set_text(f"{name}")
    st.pyplot(fig)

    output = st.empty()
    with st_capture(output.code):
        for name in list(observations.keys()):
            print(name + ': ', (observations[name][1]))


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def set_st_main():
    env = TaxiEnv(collision_sensitive_domain=True)
    return env


def play_env():
    actions_list = st.text_input("Insert actions seperated by ,:").split(',')
    domain_instance = set_st_main()

    action_dict = get_action_dictionary_from_list(domain_instance.action_index_dictionary,
                                                  domain_instance.taxis_names,
                                                  actions_list)
    obs, _, _, _ = domain_instance.step(action_dict)
    environment_snapshot(domain_instance)
    observation_snapshot(obs)


play_env()
