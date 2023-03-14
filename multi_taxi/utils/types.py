from enum import Enum, auto


class FuelType(Enum):
    FUEL = 'F'
    GAS = 'G'


class ObservationType(Enum):
    SYMBOLIC = 'symbolic'
    IMAGE = 'image'
    MIXED = 'mixed'


class SymbolicObservation(Enum):
    LOCATION_ROW = 'location_row'  # the location row of the observing taxi
    LOCATION_COL = 'location_col'  # the location column of the observing taxi
    REMAINING_FUEL = 'remaining_fuel'  # the remaining fuel of the observing taxi
    ENGINE_ON = 'engine_on'  # indicates if the observing taxi's engine is on
    IS_DEAD = 'is_dead'  # indicates if the observing taxi is dead
    OTHER_LOCATION_ROW = '{name}_location_row'  # the location row of the named taxi
    OTHER_LOCATION_COL = '{name}_location_col'  # the location column of the named taxi
    OTHER_REMAINING_FUEL = '{name}_remaining_fuel'  # the remaining fuel of the named taxi
    OTHER_ENGINE_ON = '{name}_engine_on'  # indicates if the named taxi's engine is on
    OTHER_IS_DEAD = '{name}_is_dead'  # indicates if the named taxi is dead
    PASSENGER_LOCATION_ROW = 'passenger_{index}_location_row'  # the location row of the indexed passenger
    PASSENGER_LOCATION_COL = 'passenger_{index}_location_col'  # the location column of the indexed passenger
    PASSENGER_DESTINATION_ROW = 'passenger_{index}_destination_row'  # the destination row of the indexed passenger
    PASSENGER_DESTINATION_COL = 'passenger_{index}_destination_col'  # the destination column of the indexed passenger
    PASSENGER_ARRIVED = 'passenger_{index}_arrived'  # indicates arrival of indexed passenger destination
    PASSENGER_IN_TAX = 'passenger_{p_index}_in_taxi_{t_index}'  # indicates if the indexed passenger is in the taxi
    PASSENGER_PICKED_UP = 'passenger_{index}_picked_up'  # an indicator for passenger pickup
    PASSENGER_PICKUP_ORDER = 'passenger_{index}_pickup_order'  # the position of the passenger in pickup ordering
    PASSENGER_DROPOFF_ORDER = 'passenger_{index}_dropoff_order'  # the position of the passenger in dropoff ordering


class ObstacleType(Enum):
    WALL = '|'
    BLOCKADE = 'X'


class Event(Enum):
    STEP = auto()  # performing any step
    MOVE = auto()  # perform any movement actions whether successfully completed or not
    PICKUP = auto()  # passenger was picked up
    INTERMEDIATE_DROPOFF = auto()  # drop off passenger at the wrong destination
    FINAL_DROPOFF = auto()  # drop off passenger at the final destination
    REFUEL = auto()  # successful refueling of the acting taxi
    TURN_ENGINE_ON = auto()  # engine was turned on at this step
    TURN_ENGINE_OFF = auto()  # engine was turned off at this step
    STANDBY_ENGINE_ON = auto()  # standby action performed while the engine is on
    STANDBY_ENGINE_OFF = auto()  # standby action performed while the engine is off
    USE_ENGINE_WHILE_OFF = auto()  # perform any action while requiring the engine while engine is off
    ENGINE_ALREADY_ON = auto()  # attempt to turn on the engine when it is already on
    BAD_PICKUP = auto()  # performed pickup action without a passenger present OR if taxi is full
    BAD_DROPOFF = auto()  # dropped off a passenger that is not currently in the taxi
    BAD_REFUEL = auto()  # refuel performed at location without a fuel station
    BAD_FUEL = auto()  # refuel performed at location with a fuel station of the wrong type
    USE_ENGINE_WHILE_NO_FUEL = auto()  # attempt to move while fuel capacity is at 0
    STUCK_WITHOUT_FUEL = auto()  # at current step fuel capacity dropped 0 and not on a correct fuel station
    OUT_OF_TIME = auto()  # environment has been running for more steps than the maximum steps for taxi
    HIT_OBSTACLE = auto()  # hit some obstacle in the domain map
    COLLISION = auto()  # collision occurred at this step
    DEAD = auto()  # taxi has collided in a past step and cannot act (replaces STEP)
    OBJECTIVE = auto()  # environment objective has been achieved


class Action(Enum):
    SOUTH = 'south'
    NORTH = 'north'
    EAST = 'east'
    WEST = 'west'
    PICKUP = 'pickup'
    DROPOFF = 'dropoff'
    ENGINE_ON = 'engine_on'
    ENGINE_OFF = 'engine_off'
    STANDBY = 'standby'
    REFUEL = 'refuel'


MOVE_ACTIONS = [Action.SOUTH, Action.NORTH, Action.EAST, Action.WEST]
ENGINE_ACTIONS = [Action.ENGINE_ON, Action.ENGINE_OFF]
