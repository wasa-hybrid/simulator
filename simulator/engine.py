# Engine Parts

from datetime import datetime, timedelta

from simulator.basic       import *
from simulator.parts       import *

class Engine(Part):
    ignition: datetime
    cutoff:   datetime
    thrust:   float    = 0
    mass:     float    = 1

    def setBurn(self, ignition: datetime, burntime: float):
        self.ignition = ignition
        self.cutoff   = ignition + timedelta(seconds=burntime)

    def at(self, t: datetime, env: Environment, *args) -> Q:
        q = Q(m=self.mass)
        if self.ignition <= t < self.cutoff:
            q.F = vec3(self.thrust, 0, 0)

        return q
