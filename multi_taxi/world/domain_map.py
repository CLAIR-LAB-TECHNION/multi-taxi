from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from typing import List

import numpy as np

from ..utils.types import ObstacleType, FuelType

FREE_CELL = ' '
HORIZONTAL_SEP = ':'

CELL_VALUES = {FREE_CELL, ObstacleType.BLOCKADE.value} | {fuel_type.value for fuel_type in FuelType}
SEP_VALUES = {ObstacleType.WALL.value, HORIZONTAL_SEP}


class DomainMap:
    def __init__(self, domain_map: List[str]):
        # assign domain map field for useful properties
        self.__domain_map = np.array([list(row) for row in domain_map], dtype=object)

        # assert correct map dimensions and border
        self.__border_and_dimension_check(domain_map)

        # assert correct map inner values.
        self.__inner_value_check()

        # get obstacles and fuel stations
        self.__obstacles, self.__fuel_stations = self.__get_obstacles_and_fuel_stations()

        # get list of free cells on the map.
        self.__free_locations = list(self.iter_free_locations())

    @property
    def domain_map(self):
        return self.__domain_map.copy()

    @property
    def map_height(self):
        return len(self.__domain_map) - 2  # -2 for top and bottom barriers

    @property
    def map_width(self):
        return (len(self.__domain_map[0]) - 1) // 2  # -1 for left side barriers and //2 for cell separators

    @property
    def num_locations(self):
        return self.map_width * self.map_height

    @property
    def num_free_locations(self):
        return len(list(self.iter_free_locations()))

    @property
    def obstacles(self):
        return self.__obstacles.copy()

    @staticmethod
    def location_to_domain_map_idx(row, col):
        return row + 1, 2 * col + 1

    def iter_locations(self):
        for row in range(self.map_height):
            for col in range(self.map_width):
                yield Location(row, col)

    def iter_free_locations(self):
        for loc in self.iter_locations():
            # check existence of obstacles in this location. if so, check hit without transition
            if loc not in self.__obstacles or not any(obstacle.hit(loc, loc) for obstacle in self.__obstacles[loc]):
                yield loc

    def sample_free_locations(self, n, distinct=True, rng=np.random):
        locations_idx = rng.choice(range(len(self.__free_locations)), size=n, replace=not distinct)

        # convert idx to location
        rows, cols = zip(*[self.__free_locations[i] for i in locations_idx])
        locations = np.array([rows, cols]).T

        return locations

    def get_map_value_at_location(self, location):
        map_row, map_col = self.location_to_domain_map_idx(*location)
        return self.__domain_map[map_row][map_col]

    def hit_obstacle(self, transition_start, transition_end):
        # assert hashable type
        transition_start = Location(*transition_start)
        transition_end = Location(*transition_end)

        return any(obstacle.hit(transition_start, transition_end)
                   for obstacle in self.__obstacles[transition_start] | self.__obstacles[transition_end])

    def at_fuel_station(self, location, fuel_type):
        location = Location(*location)

        return location in self.__fuel_stations and fuel_type == self.__fuel_stations[location].fuel_type

    def fuel_station_at_location(self, location):
        return self.__fuel_stations.get(location)

    def __str__(self):
        return '\n'.join(self.__domain_map)

    def __getitem__(self, x, y=None):
        if y is None:
            x, y = iter(x)
        return self.__domain_map[x][y]

    @staticmethod
    def __border_and_dimension_check(domain_map):
        assert len(domain_map) > 2, 'domain map must contain a top and bottom barrier'
        assert all(c == '-' for c in domain_map[0][1:-1]), 'domain map must contain a top barrier'
        assert all(c == '-' for c in domain_map[-1][1:-1]), 'domain map must contain a bottom barrier'
        assert domain_map[0][0] == domain_map[0][-1] == domain_map[-1][0] == domain_map[-1][-1] == '+', ('must define'
                                                                                                         'map corners'
                                                                                                         'with "+"'
                                                                                                         'character')
        assert all(len(row) > 2 for row in domain_map), 'domain map must contain left and right barriers'
        assert all(row[0] == '|' for row in domain_map[1:-1]), 'domain map must contain left barriers'
        assert all(row[-1] == '|' for row in domain_map[1:-1]), 'domain map must contain right barriers'
        assert all(len(row) == len(domain_map[0]) for row in domain_map), 'domain map must be rectangular'

    def __inner_value_check(self):
        for loc in self.iter_locations():
            v = self.get_map_value_at_location(loc)
            assert v in CELL_VALUES
            map_row, map_col = self.location_to_domain_map_idx(*loc)

            left_v = self[map_row, map_col - 1]
            assert left_v in SEP_VALUES

            right_v = self[map_row, map_col + 1]
            assert right_v in SEP_VALUES

    def __get_obstacles_and_fuel_stations(self):
        obstacles = defaultdict(set)
        fuel_stations = dict()

        # only one boundary obstacle. put multiple times in obstacles dict
        boundary_obstacle = Boundary(Location(self.map_height, self.map_width))

        # iterate all valid cell indices
        for loc in self.iter_locations():
            # get index in actual map
            value_at_loc = self.get_map_value_at_location(loc)
            map_row, map_col = self.location_to_domain_map_idx(*loc)

            # wall to the right of the current location
            if loc.col < self.map_width - 1 and self[map_row, map_col + 1] == ObstacleType.WALL.value:
                wall = Wall(loc)
                obstacles[loc].add(wall)
                obstacles[loc.next_col()].add(wall)  # save wall on both sides for symmetric query

            # blockade at the current location
            if value_at_loc == ObstacleType.BLOCKADE.value:
                obstacles[loc].add(Blockade(loc))

            # add bounds obstacle
            if any(v == 0 for v in loc) or loc.row == self.map_height - 1 or loc.col == self.map_width - 1:
                obstacles[loc].add(boundary_obstacle)

            # add fuel type to location if one exists
            if value_at_loc in [t.value for t in FuelType]:
                fuel_stations[loc] = FuelStation(loc, FuelType(value_at_loc))

        return obstacles, fuel_stations


class Location(namedtuple('Location', 'row col')):
    def next_row(self):
        return self.__class__(self.row + 1, self.col)

    def prev_row(self):
        return self.__class__(self.row - 1, self.col)

    def next_col(self):
        return self.__class__(self.row, self.col + 1)

    def prev_col(self):
        return self.__class__(self.row, self.col - 1)

    def adjacent(self):
        return [self.next_row(), self.prev_row(), self.next_col(), self.prev_col()]


class FuelStation:
    def __init__(self, location, fuel_type: FuelType):
        self.location = Location(*location)  # assert tuple type for immutability
        self.fuel_type = fuel_type

    def __str__(self):
        return f'{self.fuel_type.name} station at {self.location}'

    def __hash__(self):
        return hash((self.location, self.fuel_type.value))

    def __eq__(self, other):
        return other.location == self.location and other.fuel_type == self.fuel_type


class Obstacle(ABC):
    def __init__(self, location):
        self.location = Location(*location)  # force tuple for hashing

    @abstractmethod
    def hit(self, transition_start, transition_end) -> bool:
        pass

    def __hash__(self):
        return hash(self.location)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and other.location == self.location


class Wall(Obstacle):
    """
    An obstacle that prevents moving horizontally between adjacent cells.
    """

    def __init__(self, location):
        super().__init__(location)

        self.row = location[0]
        self.col_start = location[1]
        self.col_end = self.col_start + 1

    def hit(self, transition_start, transition_end):
        t_start_row, t_start_col = transition_start
        t_end_row, t_end_col = transition_end

        # must be on the same row
        if not t_start_row == t_end_row == self.row:
            return False

        # must be adjacent columns
        if not (t_start_col == t_end_col - 1 or t_start_col == t_end_col + 1):
            return False

        if t_start_col > t_end_col:  # check transition from left to right (symmetric)
            t_start_col, t_end_col = t_end_col, t_start_col  # swap start and end col

        return (t_start_col <= self.col_start and  # start before the wall
                t_end_col >= self.col_end)  # end after the wall


class Blockade(Obstacle):
    """
    An obstacle that prevents moving into the cell it is occupying
    """

    def hit(self, transition_start, transition_end):
        return transition_start == self.location or transition_end == self.location


class Boundary(Obstacle):
    def __init__(self, location):
        super().__init__(location)
        self.map_height, self.map_width = self.location

    def hit(self, transition_start, transition_end):
        return self.__out_of_bounds(transition_start) or self.__out_of_bounds(transition_end)

    def __out_of_bounds(self, location):
        row, col = location
        return not (0 <= row < self.map_height and 0 <= col < self.map_width)
