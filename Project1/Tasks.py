import numpy as np
import matplotlib.pyplot as plt
import assimulo.problem as apro
import matplotlib.pyplot as mpl
import assimulo.solvers as asol
import BDF4Solver
import EESolver
import BDF2Solver

# Paramters for the elastic pendulum
m = 1.0      # mass
k = 10    # spring constant
L0 = 1.0     # natural length of the spring
g = 1.0     # acceleration due to gravity


# Define Problem 
initial_conditions = [0.0, 1.1, 0.1, 0.0]  # initial position (x,y) and velocity (vx, vy)
k_list = [100, 10, 0.1]
h_list = [0.1,0.01]
EE_error_list = []
BDF4_error_list = []
BDF2_error_list = []
simulation_time = 4.9

import numpy as np
import matplotlib.pyplot as plt
from math import ceil

EE_error_list = []
BDF4_error_list = []
BDF2_error_list = []

def k_dependent_pendulum(t, y, k):
    yvec = np.zeros_like(y)
    yvec[0] = y[2]
    yvec[1] = y[3]
    r = np.sqrt(y[0]**2 + y[1]**2)
    lam = k * (r - 1) / r
    yvec[2] = -y[0] * lam
    yvec[3] = -y[1] * lam - 1
    return yvec

from scipy.interpolate import interp1d

for k in k_list:
    EE_errors_for_k = []
    BDF4_errors_for_k = []
    BDF2_errors_for_k = []

    def k_dependent_pendulum_k(t, y):
        return k_dependent_pendulum(t, y, k)

    # Reference solution with fine steps
    eP_Problem = apro.Explicit_Problem(k_dependent_pendulum_k, t0=0, y0=initial_conditions)
    eP_Solver = asol.CVode(eP_Problem)
    eP_Solver.reset()
    t_sol, y_sol = eP_Solver.simulate(simulation_time, 10000)  # fine reference

    # Interpolation functions for each component
    y_interp_funcs = [interp1d(t_sol, y_sol[:, i], kind='cubic') for i in range(y_sol.shape[1])]

    for h in h_list:
        n_steps = ceil(simulation_time / h)

        # EE Solver
        ee_solver = EESolver.Explicit_Euler(eP_Problem)
        ee_solver._set_h(h)
        ee_solver.reset()
        t, y = ee_solver.simulate(simulation_time, n_steps)

        # Interpolate reference solution at solver's time points
        y_ref = np.column_stack([f(t) for f in y_interp_funcs])
        EE_errors_for_k.append(np.linalg.norm(y_ref - y, ord=np.inf))

        # BDF4 Solver
        bdf4_solver = BDF4Solver.BDF4(eP_Problem)
        bdf4_solver._set_h(h)
        bdf4_solver.reset()
        t, y = bdf4_solver.simulate(simulation_time, n_steps)
        y_ref = np.column_stack([f(t) for f in y_interp_funcs])
        BDF4_errors_for_k.append(np.linalg.norm(y_ref - y, ord=np.inf))

        # BDF2 Solver
        bdf2_solver = BDF2Solver.BDF_2(eP_Problem)
        bdf2_solver._set_h(h)
        bdf2_solver.reset()
        t, y = bdf2_solver.simulate(simulation_time, n_steps)
        y_ref = np.column_stack([f(t) for f in y_interp_funcs])
        BDF2_errors_for_k.append(np.linalg.norm(y_ref - y, ord=np.inf))

    EE_error_list.append(EE_errors_for_k)
    BDF4_error_list.append(BDF4_errors_for_k)
    BDF2_error_list.append(BDF2_errors_for_k)
# Plotting the errors

for i, h in enumerate(h_list):
    plt.figure()
    plt.loglog(k_list, [EE_error_list[j][i] for j in range(len(k_list))], marker='o', label='EE')
    plt.loglog(k_list, [BDF4_error_list[j][i] for j in range(len(k_list))], marker='x', label='BDF4')
    plt.loglog(k_list, [BDF2_error_list[j][i] for j in range(len(k_list))], marker='s', label='BDF2')
    plt.xlabel('Spring Constant k')
    plt.ylabel('Infinity Norm of Error')
    plt.title(f'Elastic Pendulum Error vs k (h = {h})')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()
