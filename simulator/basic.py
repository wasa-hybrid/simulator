from typing import TypeVar, Type
import numpy as np
from scipy.spatial.transform import Rotation as R

def vec3(x: float, y: float, z: float):
    return np.array([x, y, z])

V3 = np.ndarray

T3 = np.ndarray

# def sixDoF(x: V3, v: V3, theta: R, omega: V3):
#     return np.concatenate([x, v, theta.as_quat(), omega])

# def sixDoFdt(v: V3, dvdt: V3, theta: R, omega: V3, domegadt: V3):
#     o1 = omega[0]
#     o2 = omega[1]
#     o3 = omega[2]
#     mat = np.array([[ 0, -o1, -o2, -o3],
#                      [o1,   0,  o3, -o2],
#                      [o2, -o3,   0,  o1],
#                      [o3,  o2, -o1,   0]])
#     dxdt     = v
#     dthetadt = np.dot(mat, theta.as_quat())
#     return np.concatenate([dxdt, dvdt, dthetadt, domegadt])

# def asSixDoF(s):
#     return [s[0:3], s[3:6], R.from_quat(s[6:10]), s[10:13]]

r0 = R.from_euler('x', 0)
v0 = vec3(0, 0, 0)
t0 = np.repeat(0, 9).reshape(3, 3)

T = TypeVar('T', bound='SixDoF') # for factory method

# 6 degrees of freedom for integral
class SixDoF:
    X:     V3 # position
    V:     V3 # velocity
    theta: R  # orientation
    omega: V3 # angular velocity

    def __init__(self, X: V3 = v0, V: V3 = v0, theta: R = r0, omega: V3 = v0):
        self.X     = X
        self.V     = V
        self.theta = theta
        self.omega = omega


    def to_values(self) -> np.ndarray:
        return np.concatenate([self.X,
                               self.V,
                               self.theta.as_quat(),
                               self.omega])

    @classmethod
    def from_values(cls: Type[T], s: np.ndarray) -> T:
        return cls(s[0:3],
                   s[3:6],
                   R.from_quat(s[6:10]),
                   s[10:13])

    @classmethod
    def dt(cls, V: V3, dVdt: V3, theta: R, omega: V3, domegadt: V3) \
        -> np.ndarray:
        o1 = omega[0]
        o2 = omega[1]
        o3 = omega[2]
        mat = np.array([[ 0, -o1, -o2, -o3],
                        [o1,   0,  o3, -o2],
                        [o2, -o3,   0,  o1],
                        [o3,  o2, -o1,   0]])
        dXdt     = V
        dthetadt = np.dot(mat, theta.as_quat())
        return np.concatenate([dXdt, dVdt, dthetadt, domegadt])

# Quantities for equation of motion
class Q:
    m: float # mass
    I: T3    # moment of inertia
    F: V3    # external force
    M: T3    # moment of force

    def __init__(self, m: float = 0, I: T3 = t0, F: V3 = v0, M: T3 = t0):
        self.m = m
        self.I = I
        self.F = F
        self.M = M
