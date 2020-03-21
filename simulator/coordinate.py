# Functions for converting between coordinate systems

import math
from datetime import datetime, timedelta
import numpy   as np
import pymap3d as pm
from astropy.time import Time

from simulator.basic import *
from simulator.environment import *

earth_rotation             = 7.2921159e-5

# def body2eci(X: V3, theta: R):
    # return theta.inv().apply(X)

def eci2ecef(X: V3, t: datetime) -> V3:
    return np.array(pm.eci2ecef(*X, t))
    # return R.from_euler('z', env.earth_rotation * t).apply(X)

def ecef2eci(X: V3, t: datetime) -> V3:
    return np.array(pm.ecef2eci(*X, t))[:, 0]
    # return R.from_euler('z', - env.earth_rotation * t).apply(X)

def v_eci2ecef(V: V3, t: datetime, Xeci: V3) -> V3:
    omega = earth_rotation
    return eci2ecef(V, t) - vec3(- omega * Xeci[1], omega * Xeci[0], 0)

def v_ecef2eci(V: V3, t: datetime, Xeci: V3) -> V3:
    omega = earth_rotation
    return ecef2eci(V, t) + vec3(- omega * Xeci[1], omega * Xeci[0], 0)

def ecef2llh(X: V3) -> V3:
    return np.array(pm.ecef2geodetic(*X, deg=False))
    # x = X[0]
    # y = X[1]
    # z = X[2]
    # p     = math.sqrt(x ** 2 + y ** 2)
    # theta = math.atan(y / x)
    # a     = env.earth_ellipsoid_semimajor
    # b     = env.earth_ellipsoid_semiminor
    # e2    = env.earth_ellipsoid_eccentricity2
    # ed2   = env.earth_ellipsoid_eccentricity_s2
    # phi = math.atan((z + ed2 * b * (math.sin(theta) ** 3)) / p - e2 * a * (math.cos(theta) ** 3))
    # lam = atan(y / x)
    # N   = a / math.sqrt(1 - e2 * (math.sin(phi) ** 2))
    # h   = (p / math.cos(phi)) - N
    # return vec3(phi, lam, h)

def llh2ecef(X: V3) -> V3:
    return np.array(pm.geodetic2ecef(*X, deg=False))
    # phi = X[0]
    # lam = X[1]
    # h   = X[2]
    # a  = env.earth_ellipsoid_semimajor
    # f  = env.earth_ellipsoid_flattening
    # e2 = env.earth_ellipsoid_eccentricity2
    # N  = a / math.sqrt(1 - e2 * (math.sin(phi) ** 2))
    # cos_phi = math.cos(phi)
    # cos_lam = math.cos(lam)
    # sin_phi = math.sin(phi)
    # sin_lam = math.sin(lam)
    # x  = (N + h) * cos_phi * cos_lam
    # y  = (N + h) * cos_phi * sin_lam
    # z  = (N * (1 - e2) + h) * sin_phi
    # return vec3(x, y, z)

def ecef2enu(X: V3, Ollh: V3) -> V3:
    return np.array(pm.ecef2enu(*X, *Ollh, deg=False))

def enu2ecef(X: V3, Ollh: V3) -> V3:
    return np.array(pm.enu2ecef(*X, *Ollh, deg=False))

def ecef2aer(X: V3, Ollh: V3) -> V3:
    return np.array(pm.ecef2aer(*X, *Ollh, deg=False))

def aer2ecef(X: V3, Ollh: V3) -> V3:
    return np.array(pm.aer2ecef(*X, *Ollh, deg=False))

def v_aer2ecef(V: V3, Ollh: V3) -> V3:
    return aer2ecef(V, Ollh) - llh2ecef(Ollh)

def body2ned(X: V3, theta: R) -> V3:
    return theta.inv().apply(X)

def ned2ecef(X: V3, Ollh: V3) -> V3:
    return np.array(pm.ned2ecef(*X, *Ollh, deg=False))


def body2ecef(X: V3, theta: R, Oecef: V3) -> V3:
    return ned2ecef(body2ned(X, theta), ecef2llh(Oecef))

def body2eci(X: V3, theta: R, Oeci: V3, t: datetime) -> V3:
    return ecef2eci(body2ecef(X, theta, eci2ecef(Oeci, t)), t)

def llh2r_ecef(X: V3) -> R:
    return R.from_euler('yz', [-X[0], X[1]])

def v_ned2ecef(X: V3, Ollh: V3) -> V3:
    print('r', llh2r_ecef(Ollh).as_rotvec())
    return llh2r_ecef(Ollh).inv().apply(R.from_euler('y', math.pi/2).apply(X))

def r_ned2ecef(theta: R, Ollh: V3) -> R:
    print('r', R.from_euler('yz', [-math.pi/2 - Ollh[0], Ollh[1]]).as_rotvec())
    print('t', theta.as_rotvec())
    return R.from_euler('yz', [-math.pi/2 - Ollh[0], Ollh[1]]) * theta
    # return R.from_rotvec(v_ned2ecef(theta.as_rotvec(), Ollh))
    # return theta * llh2r_ecef(Ollh).inv()

def r_ecef2eci(theta: R, t: datetime) -> R:
    gst = Time(t).sidereal_time("apparent", "greenwich").radian
    gr  = R.from_euler('z', gst)
    return gr * theta

def a_body2eci(X: V3, theta: R, t: datetime) -> V3:
    return theta.apply(X)
    # return r_ecef2eci(theta, t).apply(X)
