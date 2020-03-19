import numpy as np
from scipy.spatial.transform import Rotation as R

def vec3(x: float, y: float, z: float):
    return np.array([x, y, z])

Vec3 = np.ndarray

def sixDoF(x: Vec3, v: Vec3, theta: R, omega: R):
    return np.concatenate([x, v, theta.as_quat(), omega.as_quat()])

def asSixDoF(s):
    return [s[0:3], s[3:6], R.from_quat(s[6:10]), R.from_quat(s[10:14])]

class SixDoF:
    def __init__(self, x: Vec3, v: Vec3, theta: R, omega: R):
        self.x     = x
        self.v     = v
        self.theta = theta
        self.omega = omega


r0 = R.from_euler('x', 0)
v0 = vec3(0, 0, 0)
