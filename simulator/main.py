from datetime import datetime, timedelta
import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
import astropy.time as at

from simulator.basic       import *
from simulator.coordinate  import *
from simulator.environment import *
from simulator.parts       import *
from simulator.engine      import *

def func(ti: float, s, args):
    env, body = args
    t = env.time(ti)
    S = SixDoF.from_values(s)
    Xeci  = S.X
    Veci  = S.V
    theta = S.theta
    omega = S.omega

    Sbody = S

    q = body.at(t, env)

    # print(body2eci(q.F / q.m, theta, v0, t))
    # dVdt = body2eci(q.F / q.m, theta, v0, t) + env.gravity(Xeci)
    dVdt = a_body2eci(q.F / q.m, theta, t) + env.gravity(Xeci)
    domegadt = v0
    return SixDoF.dt(Veci, dVdt, theta, omega, domegadt)

def main():
    print("WASA Rocket Simulator\n")

    t0 = datetime.now()

    X0llh = vec3(np.radians(35.7058879), np.radians(139.7060483), 0)
    X0eci = ecef2eci(llh2ecef(X0llh), t0)

    V0aer = vec3(np.radians(30), np.radians(70), 100)
    V0eci = v_ecef2eci(v_aer2ecef(V0aer, X0llh), t0, X0eci)


    R0ned = R.from_euler('y', np.radians(120))

    R0 = R0ned
    print(R0.as_rotvec())
    print('n', r_ned2ecef(R0ned, X0llh).as_rotvec())
    R0    = r_ecef2eci(r_ned2ecef(R0ned, X0llh), t0)
    print(R0.as_rotvec())

    s0 = SixDoF(X0eci, V0eci, R0, v0)

    engine = Engine()
    engine.thrust = 40
    engine.mass = 0.5
    engine.setBurn(t0 + timedelta(seconds=4), 2)

    body = engine

    t_max = 40.0
    dt = 0.1

    # ts = np.arange(0.0, t_max, dt)
    # dts = list(map(lambda t: t0 + timedelta(seconds=t), ts.tolist()))

    env    = Environment()
    env.t0 = t0

    XS  = np.zeros([int(t_max/dt), 13])
    dts = [None] * (int(t_max/dt))

    solver = ode(func)
    solver.set_integrator('dop853')
    solver.set_initial_value(s0.to_values())
    solver.set_f_params((env, body))


    print("running...")
    i = 0
    while solver.successful() and solver.t < t_max:
        solver.integrate(solver.t + dt)
        XS[i]  = solver.y
        dts[i] = t0 + timedelta(seconds=solver.t)
        i += 1

    print("finished.")

    # sol = odeint(func, s0.to_values(), ts, args=(env, body))


    XSeci  = XS[:, 0:3].T
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
