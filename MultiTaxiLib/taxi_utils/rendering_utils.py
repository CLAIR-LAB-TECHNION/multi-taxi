"""
This file encapsulates all relevant rendering functions and their utils (both grid and image rendering).
"""

from typing import Optional
import numpy as np
from MultiTaxiLib.config import COLOR_MAP
import matplotlib.pyplot as plt
from gym.utils import colorize
from io import StringIO
import sys
from contextlib import closing


def map2rgb(state: list, np_map: np.ndarray) -> Optional[np.array]:
    """
    Given a numpy-ascii map - return an image formatted map.
    Args:
        state: current state of the domain
        np_map: numpy ascii map

    Returns: np array rgb map

    """
    if np_map is None:
        return None
    m, n = np_map.shape[0], np_map.shape[1]
    taxis_locations, _, passengers_start_locations, destinations, passengers_status = state

    for i, location in enumerate(taxis_locations):
        np_map[location[0] + 1, location[1] * 2 + 1] = str(i + 1)

    for i, location in enumerate(passengers_start_locations):
        if location not in taxis_locations:
            if passengers_status[i] > 2 or passengers_status[i] == 1:
                np_map[location[0] + 1, location[1] * 2 + 1] = ':'
            else:
                np_map[location[0] + 1, location[1] * 2 + 1] = 'P' + str(i)

    for i, location in enumerate(destinations):
        if location not in taxis_locations:
            if passengers_status[i] == 1:
                np_map[location[0] + 1, location[1] * 2 + 1] = 'P' + str(i)
            else:
                np_map[location[0] + 1, location[1] * 2 + 1] = 'D' + str(i)

    rgb_arr = np.zeros((m, n, 3), dtype=int)
    for i in range(m):
        for j in range(n):
            try:
                rgb_arr[i, j] = COLOR_MAP[np_map[i, j]]
            except KeyError:
                rgb_arr[i, j] = COLOR_MAP[np_map[i, j].decode('utf-8')]

    return rgb_arr


def plot_window_of_observation(view_len: int, taxis_locations: list) -> None:
    """
    Plots window of observation around all taxis in red.
    Args:
        view_len: height and width diameter of the window.
        taxis_locations: list of taxis current coordinates.

    Returns: Nothing

    """
    view_len += 1
    for location in taxis_locations:
        i, j = location[0] + 1, location[1] * 2 + 1
        window_size = view_len * 2 + 1
        plt.plot(range(j - view_len, j + view_len + 1), [i - view_len] * window_size, color='red')
        plt.plot(range(j - view_len, j + view_len + 1), [i + view_len] * window_size, color='red')
        plt.plot([j - view_len] * window_size, range(i - view_len, i + view_len + 1), color='red')
        plt.plot([j + view_len] * window_size, range(i - view_len, i + view_len + 1), color='red')
    view_len -= 1


def get_current_map_with_agents(domain_map: list, state: list, num_taxis: int, collided: np.ndarray) -> np.ndarray:
    """
    Returns the current (with agents movements) asci map in the numpy format
    Args:
        domain_map: Domain map in a grid format
        state: the current state of the environment (includes agents positions etc)
        num_taxis: number of taxis in the environment
        collided: collision status of all taxis

    Returns: Returns the current (with agents movements) asci map in the numpy format

    """

    # Copy map to work on

    out = domain_map.copy()
    out = [[c.decode('utf-8') for c in line] for line in out]

    taxis, fuels, passengers_start_coordinates, destinations, passengers_locations = state

    colors = ['yellow', 'red', 'white', 'green', 'cyan', 'crimson', 'gray', 'magenta'] * 5
    colored = [False] * num_taxis

    def ul(x):
        """returns underline instead of spaces when called"""
        return "_" if x == " " else x

    for i, location in enumerate(passengers_locations):
        if location > 2:  # Passenger is on a taxi
            taxi_row, taxi_col = taxis[location - 3]

            # Coloring taxi's coordinate on the map
            out[1 + taxi_row][2 * taxi_col + 1] = colorize(
                out[1 + taxi_row][2 * taxi_col + 1], colors[location - 3], highlight=True, bold=True)
            colored[location - 3] = True
        else:  # Passenger isn't in a taxi
            # Coloring passenger's coordinates on the map
            pi, pj = passengers_start_coordinates[i]
            out[1 + pi][2 * pj + 1] = colorize(out[1 + pi][2 * pj + 1], 'green', bold=True)
            colored[location - 2] = True

    for i, taxi in enumerate(taxis):
        if collided[i] == 0:  # Taxi isn't collided
            taxi_row, taxi_col = taxi
            out[1 + taxi_row][2 * taxi_col + 1] = colorize(
                ul(out[1 + taxi_row][2 * taxi_col + 1]), colors[i], highlight=True)
        else:  # Collided!
            taxi_row, taxi_col = taxi
            out[1 + taxi_row][2 * taxi_col + 1] = colorize(
                ul(out[1 + taxi_row][2 * taxi_col + 1]), 'gray', highlight=True)

    for dest in destinations:
        di, dj = dest
        out[1 + di][2 * dj + 1] = colorize(out[1 + di][2 * dj + 1], 'green')

    return np.array(out)


def render(domain_map: list, state: list, num_taxis: int, collided: np.ndarray, last_action: dict,
           action_index_dictionary: dict, dones: dict, mode: str = 'human') -> str:
    """
    Renders the domain map at the current state
    Args:
        domain_map: Domain map in a grid format
        state: the current state of the environment (includes agents positions etc)
        num_taxis: number of taxis in the environment
        collided: collision status of all taxis
        last_action: dictionary with the last action each agent took.
        action_index_dictionary: {action: str -> index: int} dictionary
        dones: a dictionary indicating for each agent if it's done
        mode: Demand mode (file or human watching).

    Returns: Value string of writing the output (grid world int string world).

    """
    outfile = StringIO() if mode == 'ansi' else sys.stdout

    out = get_current_map_with_agents(domain_map, state, num_taxis, collided)
    outfile.write("\n".join(["".join(row) for row in out]) + "\n")

    taxis, fuels, passengers_start_coordinates, destinations, passengers_locations = state
    colors = ['yellow', 'red', 'white', 'green', 'cyan', 'crimson', 'gray', 'magenta'] * 5

    # Rendering actions and passengers/ taxis status

    if last_action is not None:
        # moves = ALL_ACTIONS_NAMES
        moves = list(action_index_dictionary.keys())
        output = [moves[i] for i in np.array(list(last_action.values())).reshape(-1)]
        outfile.write("  ({})\n".format(' ,'.join(output)))
    for i, taxi in enumerate(taxis):
        outfile.write("Taxi{}-{}: Fuel: {}, Location: ({},{}), Collided: {}\n".format(i + 1, colors[i].upper(),
                                                                                      fuels[i], taxi[0], taxi[1],
                                                                                      collided[i] == 1))
    for i, location in enumerate(passengers_locations):
        start = tuple(passengers_start_coordinates[i])
        end = tuple(destinations[i])
        if location == 1:
            outfile.write("Passenger{}: Location: Arrived!, Destination: {}\n".format(i + 1, end))
        if location == 2:
            outfile.write("Passenger{}: Location: {}, Destination: {}\n".format(i + 1, start, end))
        else:
            outfile.write("Passenger{}: Location: Taxi{}, Destination: {}\n".format(i + 1, location - 2, end))
    outfile.write("Done: {}, {}\n".format(all(dones.values()), dones))
    outfile.write("Passengers Status's: {}\n".format(state[-1]))

    # No need to return anything for human
    if mode != 'human':
        with closing(outfile):
            return outfile.getvalue()


def parse_fuel_stations_from_map(map_array: np.ndarray) -> list:
    """
    Returns the locations of fuel stations from a given map.
    Args:
        map_array: the environment map

    Returns: list with locations ([i, j]) of the fuel stations over the map.

    """
    fuel_stations = []
    for i, row in enumerate(map_array[1:-1]):
        for j, char in enumerate(row[1:-1:2]):
            loc = [i, j]
            # if char == b'X':
            #     self.passengers_locations.append(loc)
            if char == b'F' or char == b'G':
                fuel_stations.append(loc)

    return fuel_stations
