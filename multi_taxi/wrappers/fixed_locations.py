from abc import ABC, abstractmethod

from pettingzoo.utils import BaseParallelWraper


class FixedLocationsWrapper(BaseParallelWraper, ABC):
    def __init__(self, env, *locs):
        super().__init__(env)

        self.locs = [
            (locs[i], locs[i + 1])
            for i in range(0, len(locs), 2)
        ]

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.set_locations()

        # must re-observe with new locations
        obs = self.unwrapped.observe_all()

        return obs

    @abstractmethod
    def set_locations(self):
        pass

    def seed(self, seed=None):
        return self.env.seed(seed)


class FixedPassengerStartLocationsWrapper(FixedLocationsWrapper):
    def set_locations(self):
        s = self.env.state()

        for p, loc in zip(s.passengers, self.locs):
            p.location = loc

        self.unwrapped.set_state(s)


class FixedPassengerDestinationsWrapper(FixedLocationsWrapper):
    def set_locations(self):
        s = self.state()
        for p, loc in zip(s.passengers, self.locs):
            p.destination = loc

        self.unwrapped.set_state(s)


class FixedTaxiStartLocationsWrapper(FixedLocationsWrapper):
    def set_locations(self):
        s = self.state()
        for t, loc in zip(s.taxis, self.locs):
            t.location = loc

        self.unwrapped.set_state(s)
