from copy import copy, deepcopy


class MultiTaxiEnvState:
    def __init__(self, taxis, passengers):
        self.taxis = taxis
        self.passengers = passengers

        self.__taxi_map = {taxi.name: taxi for taxi in taxis}

    @property
    def taxi_by_name(self):
        return self.__taxi_map

    def copy(self):
        return deepcopy(self)

    def __copy__(self):
        return self.__class__(copy(self.taxis), copy(self.passengers))

    def __deepcopy__(self, memodict={}):
        copy_ = self.__class__(deepcopy(self.taxis, memodict), deepcopy(self.passengers, memodict))

        memodict[id(self)] = copy_

        return copy_

    def __hash__(self):
        return hash((tuple(self.taxis), tuple(self.passengers)))

    def __eq__(self, other):
        return self.taxis == other.taxis and self.passengers == other.passengers
