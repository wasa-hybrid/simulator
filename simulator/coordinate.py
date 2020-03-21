# Functions for converting between coordinate systems

import math
from datetime import datetime, timedelta
import numpy   as np
import pymap3d as pm

from simulator.basic import *
from simulator.environment import *

earth_rotation             = 7.2921159e-5

# def body2eci(X: Vec3, theta: R):
    # return theta.inv().apply(X)

def eci2ecef(X: Vec3, t: datetime):
    # t2 = list(map(lambda ti: ti + timedelta(seconds=0.5), t))
    # t3 = list(map(lambda ti: ti - timedelta(seconds=0.5), t))
    # print(X, t)
    return np.array(pm.eci2ecef(X[0], X[1], X[2], t))
            # np.array(pm.eci2ecef(X[0], X[1], X[2], t2)) + \
            # np.array(pm.eci2ecef(X[0], X[1], X[2], t3))) / 3.
    # return R.from_euler('z', env.earth_rotation * t).apply(X)

def ecef2eci(X: Vec3, t: datetime):
    return np.array(pm.ecef2eci(X[0], X[1], X[2], t))[:, 0]
    # return R.from_euler('z', - env.earth_rotation * t).apply(X)

def v_eci2ecef(V: Vec3, t: datetime, Xeci: Vec3):
    omega = earth_rotation
    return eci2ecef(V, t) - vec3(- omega * Xeci[1], omega * Xeci[0], 0)

def v_ecef2eci(V: Vec3, t: datetime, Xeci: Vec3):
    omega = earth_rotation
    return ecef2eci(V, t) + vec3(- omega * Xeci[1], omega * Xeci[0], 0)

def ecef2llh(X: Vec3):
    return np.array(pm.ecef2geodetic(X[0], X[1], X[2], deg=False))
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

def llh2ecef(X: Vec3):
    return np.array(pm.geodetic2ecef(X[0], X[1], X[2], deg=False))
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

def ecef2enu(X: vec3, Ollh: vec3):
    return np.array(pm.ecef2enu(X[0], X[1], X[2], Ollh[0], Ollh[1], Ollh[2],
                                deg=False))

def enu2ecef(X: vec3, Ollh: vec3):
    return np.array(pm.enu2ecef(X[0], X[1], X[2], Ollh[0], Ollh[1], Ollh[2],
                                deg=False))

def ecef2aer(X: vec3, Ollh: vec3):
    return np.array(pm.ecef2aer(X[0], X[1], X[2], Ollh[0], Ollh[1], Ollh[2],
                                deg=False))

def aer2ecef(X: vec3, Ollh: vec3):
    return np.array(pm.aer2ecef(X[0], X[1], X[2], Ollh[0], Ollh[1], Ollh[2],
                                deg=False))

def v_aer2ecef(V: Vec3, Ollh: Vec3):
    return aer2ecef(V, Ollh) - llh2ecef(Ollh)
