# Parts

from abc import ABCMeta, abstractmethod
from datetime import datetime, timedelta

from simulator.basic       import *
from simulator.environment import *

class Part(metaclass=ABCMeta):
    origin: V3 = v0

    @abstractmethod
    def at(self, t: datetime, env: Environment, *args,
           Vbody: V3) -> Q:
        pass
