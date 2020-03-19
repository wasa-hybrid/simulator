import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D

from simulator.basic import *

def func(s, t, r, m):
    x, v, theta, omega = asSixDoF(s)

    g = 9.80665

    dxdt = v
    dvdt = vec3(0, 0, (-r * v[2] - m * g) / m)
    domegadt = r0
    dthetadt = omega
    return sixDoF(dxdt, dvdt, dthetadt, domegadt)

def main():
    print("WASA Rocket Simulator\n")

    s0 = sixDoF(v0, vec3(1, 2, 70), r0, r0)

    r = 12
    m = 100
    t = np.arange(0, 20, 0.1)

    sol = odeint(func, s0, t, args=(r, m))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sol[:, 0], sol[:, 1], sol[:, 2])
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_zlim(0, 200)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    # plt.plot(t, sol[:, 0], 'b', label='x')
    # plt.plot(t, sol[:, 1], 'g', label='v')
    # plt.legend(loc='best')
    # plt.xlabel('t')
    # plt.grid()
    # plt.show()

if __name__ == "__main__":
    main()
