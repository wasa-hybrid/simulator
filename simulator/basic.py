import numpy as np
from scipy.spatial.transform import Rotation as R

def vec3(x: float, y: float, z: float):
    return np.array([x, y, z])

Vec3 = np.ndarray

def sixDoF(x: Vec3, v: Vec3, theta: R, omega: Vec3):
    return np.concatenate([x, v, theta.as_quat(), omega])

def sixDoFdt(v: Vec3, dvdt: Vec3, theta: R, omega: Vec3, domegadt: Vec3):
    o1 = omega[0]
    o2 = omega[1]
    o3 = omega[2]
    mat = np.array([[ 0, -o1, -o2, -o3],
                     [o1,   0,  o3, -o2],
                     [o2, -o3,   0,  o1],
                     [o3,  o2, -o1,   0]])
    dxdt     = v
    dthetadt = np.dot(mat, theta.as_quat())
    return np.concatenate([dxdt, dvdt, dthetadt, domegadt])

def asSixDoF(s):
    return [s[0:3], s[3:6], R.from_quat(s[6:10]), s[10:13]]

r0 = R.from_euler('x', 0)
v0 = vec3(0, 0, 0)

class SixDoF:
    x     = v0 # position
    v     = v0 # velocity
    theta = r0 # orientation
    omega = v0 # angular velocity

