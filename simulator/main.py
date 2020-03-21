from datetime import datetime, timedelta
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D

from simulator.basic       import *
from simulator.coordinate  import *
from simulator.environment import *
from simulator.engine      import *

def func(s, t, env):
    Xeci, Veci, theta, omega = asSixDoF(s)


    g = 9.80665

    dVdt = env.gravity(Xeci)
    domegadt = v0
    return sixDoFdt(Veci, dVdt, theta, omega, domegadt)

def main():
    print("WASA Rocket Simulator\n")

    t0 = datetime.now()

    X0llh = vec3(np.radians(35.7058879), np.radians(139.7060483), 0)
    X0eci = ecef2eci(llh2ecef(X0llh), t0)

    V0aer = vec3(np.radians(0), np.radians(70), 100)
    V0eci = v_ecef2eci(v_aer2ecef(V0aer, X0llh), t0, X0eci)

    s0 = sixDoF(X0eci, V0eci, r0, v0)

    engine = Engine()

    t_max = 20
    t_step = 0.1

    ts = np.arange(0, t_max, t_step)
    dts = list(map(lambda t: t0 + timedelta(seconds=t), ts.tolist()))

    # print(list(map (lambda t: t.strftime('%Y-%m-%dT%H:%M:%S.%f%z') ,dts)))

    env = Environment()

    sol = odeint(func, s0, ts, args=(env,))


    XSeci  = sol[:, 0:3].T
    XSecef = eci2ecef(XSeci, dts)
    # XSenu = XSecef
    XSenu  = ecef2enu(XSecef, X0llh)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = XSenu[0]
    ys = XSenu[1]
    zs = XSenu[2]
    ax.plot(xs, ys, zs)
    # scale = np.array([xs.max(), -xs.min(),
    #                   ys.max(), -ys.min(),
    #                   zs.max(), -zs.min()]).max()
    # ax.set_xlim(-scale, scale)
    # ax.set_ylim(-scale, scale)
    # ax.set_zlim(-scale, scale)
    max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() * 0.5

    mid_x = (xs.max()+xs.min()) * 0.5
    mid_y = (ys.max()+ys.min()) * 0.5
    mid_z = (zs.max()+zs.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

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
