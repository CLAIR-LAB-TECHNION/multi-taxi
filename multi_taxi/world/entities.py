from copy import copy, deepcopy

from ..utils.types import Action

PASSENGER_NOT_IN_TAXI = -1
COLORS = ['yellow', 'red', 'white', 'green', 'cyan', 'blue', 'magenta']


class Taxi:
    """
    The object representation of a taxi
    """

    def __init__(self, id_, location, max_capacity, fuel, max_fuel, fuel_type, n_steps, max_steps, passengers=None,
                 can_collide=False, collided=False, engine_on=True):
        self.id = id_
        self.location = tuple(location)  # assert tuple type for immutability
        self.max_capacity = max_capacity
        self.fuel = fuel
        self.max_fuel = max_fuel
        self.fuel_type = fuel_type
        self.n_steps = n_steps
        self.max_steps = max_steps

        if passengers is None:
            self.passengers = set()
        else:
            self.passengers = set(passengers)  # assert set type for fast membership check

        self.can_collide = can_collide
        self.collided = collided
        self.engine_on = engine_on

        # set taxi color for rendering
        self.color = COLORS[self.id % len(COLORS)]

    @property
    def name(self):
        return f'taxi_{self.id}'

    def pick_up(self, passenger):
        assert not self.is_full, 'cannot pick up any passengers when the taxi is full'
        assert passenger.location == self.location, 'cannot pick up a passenger that is not at the taxi\'s location'

        self.passengers.add(passenger)
        passenger.picked_up(self.id)

    def drop_off(self, passenger):
        self.passengers.remove(passenger)
        passenger.dropped_off()

    def drop_all(self):
        for p in self.passengers:
            p.dropped_off()
        self.passengers = {}

    def move(self, direction: Action, simulation=False):
        row, col = self.location

        if direction == Action.SOUTH.value:
            row += 1
        elif direction == Action.NORTH.value:
            row -= 1
        elif direction == Action.EAST.value:
            col += 1
        elif direction == Action.WEST.value:
            col -= 1
        else:
            raise NotImplemented(f'unsupported direction {direction.value}')

        new_loc = (row, col)

        if simulation:
            return new_loc
        else:
            assert not self.empty_tank, 'cannot move with an empty fuel tank'

            # move taxi one step to the given direction
            self.location = new_loc

            # move taxi passengers one step to the given direction
            for passenger in self.passengers:
                passenger.location = new_loc

            # reduce fuel capacity by 1
            self.fuel -= 1

    @property
    def empty_tank(self):
        return self.fuel == 0

    @property
    def is_full(self):
        return self.capacity == self.max_capacity

    @property
    def out_of_time(self):
        return self.n_steps >= self.max_steps

    @property
    def capacity(self):
        return len(self.passengers)

    @property
    def engine_off(self):
        return not self.engine_on

    def toggle_engine(self):
        self.engine_on = not self.engine_on

    def refuel(self, fill=None):
        if fill is None:
            fill = self.max_fuel

        self.fuel = min(self.max_fuel, self.fuel + fill)

    @classmethod
    def from_lists(cls, ids, locations, capacities, fuels, max_fuels, fuel_types, taxi_n_steps, taxi_max_steps,
                   taxi_passengers, taxis_can_collide, taxis_collided, engines_on):
        taxis = []
        for i in range(len(ids)):
            id_ = ids[i]
            location = locations[i]
            capacity = capacities[i]
            fuel = fuels[i]
            max_fuel = max_fuels[i]
            fuel_type = fuel_types[i]
            n_steps = taxi_n_steps[i]
            max_steps = taxi_max_steps[i]
            passengers = taxi_passengers[i]
            can_collide = taxis_can_collide[i]
            collided = taxis_collided[i]
            engine_on = engines_on[i]

            taxis.append(cls(id_, location, capacity, fuel, max_fuel, fuel_type, n_steps, max_steps, passengers,
                             can_collide, collided, engine_on))

        return taxis

    def __contains__(self, item):
        return item in self.passengers

    def __str__(self):
        fuel_str = f'{self.fuel}/{self.max_fuel}' if self.max_fuel != float('inf') else f'{self.fuel}'
        step_str = f'{self.n_steps}/{self.max_steps}' if self.max_steps != float('inf') else f'{self.n_steps}'

        return (f'Taxi{self.id}-{self.color.upper()}: Fuel: {fuel_str}, Location: {self.location}, '
                f'Engine: {"ON" if self.engine_on else "OFF"}, Collided: {self.collided}, Step: {step_str}')

    def __copy__(self):
        return self.__class__(self.id,
                              self.location,
                              self.max_capacity,
                              self.fuel,
                              self.max_fuel,
                              self.fuel_type,
                              self.n_steps,
                              self.max_steps,
                              copy(self.passengers),
                              self.can_collide,
                              self.collided,
                              self.engine_on)

    def __deepcopy__(self, memodict={}):
        copy_ = self.__class__(self.id,
                               self.location,
                               self.max_capacity,
                               self.fuel,
                               self.max_fuel,
                               self.fuel_type,
                               self.n_steps,
                               self.max_steps,
                               deepcopy(self.passengers, memodict),
                               self.can_collide,
                               self.collided,
                               self.engine_on)
        memodict[id(self)] = copy_

        return copy_

    def __hash__(self):
        return hash((self.id, self.location, self.capacity, self.fuel, self.max_fuel, self.fuel_type, self.n_steps,
                     self.max_steps, tuple(sorted(self.passengers, key=lambda p: p.id)), self.can_collide,
                     self.collided, self.engine_on))

    def __eq__(self, other):
        return (self.id == other.id and
                self.location == other.location and
                self.capacity == other.capacity and
                self.fuel == other.fuel and
                self.max_fuel == other.max_fuel and
                self.fuel_type == other.fuel_type and
                self.n_steps == other.n_steps and
                self.max_steps == other.max_steps and
                self.passengers == other.passengers and
                self.can_collide == other.can_collide and
                self.collided == other.collided and
                self.engine_on == other.engine_on)


class Passenger:
    def __init__(self, id_, location, destination, carrying_taxi=PASSENGER_NOT_IN_TAXI):
        self.id = id_
        self.location = tuple(location)  # assert tuple type for immutability
        self.destination = tuple(destination)
        self.carrying_taxi = carrying_taxi

        # set passenger color for rendering
        self.color = COLORS[self.id % len(COLORS)]

    @property
    def arrived(self):
        return self.location == self.destination and not self.in_taxi

    @property
    def in_taxi(self):
        return self.carrying_taxi != PASSENGER_NOT_IN_TAXI

    def picked_up(self, taxi_id):
        self.carrying_taxi = taxi_id

    def dropped_off(self):
        self.carrying_taxi = PASSENGER_NOT_IN_TAXI

    @classmethod
    def from_list(cls, ids, locations, destinations, carrying_taxis):
        passengers = []
        for i in range(len(ids)):
            id_ = ids[i]
            location = locations[i]
            destination = destinations[i]
            carrying_taxi = carrying_taxis[i]

            passengers.append(cls(id_, location, destination, carrying_taxi))

        return passengers

    def __str__(self):
        cur_loc = self.location
        dest_loc = self.destination
        if self.in_taxi:
            cur_loc = f'Taxi{self.carrying_taxi} {cur_loc}'
        if self.arrived:
            dest_loc = f'Arrived! {dest_loc}'

        return f'Passenger{self.id}-{self.color.upper()}: Location: {cur_loc}, Destination: {dest_loc}'

    def __hash__(self):
        return hash((self.id, self.location, self.destination, self.carrying_taxi))

    def __eq__(self, other):
        return (self.id == other.id and self.location == other.location and self.destination == other.destination and
                self.carrying_taxi == other.carrying_taxi)

    def __copy__(self):
        return self.__class__(self.id,
                              self.location,
                              self.destination,
                              self.carrying_taxi)

    def __deepcopy__(self, memodict={}):
        copy_ = copy(self)
        memodict[id(self)] = copy_

        return copy_
