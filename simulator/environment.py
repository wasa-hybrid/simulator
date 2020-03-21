# Environments

import math
from datetime import datetime, timedelta

from simulator.basic       import *

class Environment:
    # Simulation settings
    t0: datetime

    # Constants
    g  = 9.80619920
    GM = 3.986004418e14

    # WGS84
    earth_ellipsoid_flattening = 1.0 / 298.257223563
    earth_ellipsoid_semimajor  = 6378137.0
    earth_ellipsoid_semiminor  = earth_ellipsoid_semimajor - earth_ellipsoid_flattening * earth_ellipsoid_semimajor
    earth_ellipsoid_eccentricity2 = (earth_ellipsoid_semimajor ** 2 - earth_ellipsoid_semiminor ** 2) / earth_ellipsoid_semimajor ** 2
    earth_ellipsoid_eccentricity_s2 = (earth_ellipsoid_semimajor ** 2 - earth_ellipsoid_semiminor ** 2) / earth_ellipsoid_semiminor ** 2

    def time(self, t: float) -> datetime:
        return self.t0 + timedelta(seconds=t)

    def gravity(self, Xeci: V3) -> V3:
        R = np.linalg.norm(Xeci)
        return - self.GM / (R ** 3) * Xeci

    def gravity_surface(self, Xeci: V3) -> V3:
        return - self.g / np.linalg.norm(Xeci) * Xeci


