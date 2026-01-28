import numpy as np
import matplotlib.pyplot as plt
import assimulo.problem as apro
import matplotlib.pyplot as mpl
import assimulo.solvers as asol
import BDF4Solver
import BDF2Solver 

# Paramters for the elastic pendulum
m = 1.0      # mass
k = 10    # spring constant
L0 = 1.0     # natural length of the spring
g = 1.0     # acceleration due to gravity

def elastic_pendulum(t,y):
    yvec = np.zeros_like(y)
    yvec[0] = y[2]
    yvec[1] = y[3]
    lam = k * (np.sqrt(y[0]**2 + y[1]**2) - 1) / np.sqrt(y[0]**2 + y[1]**2)
    yvec[2] = -1*y[0]*lam
    yvec[3] = -1*y[1]*lam - 1
    return yvec

# Define Problem 
initial_conditions = [0.0, 1.1, 0.1, 0.0]  # initial position (x,y) and velocity (vx, vy)
eP_Problem = apro.Explicit_Problem(elastic_pendulum, t0 = 0, y0 = initial_conditions)
eP_Problem.name = r'Elastic Pendulum (m={m}, k={k}, L0={L0}, g={g})'
eP_Problem.name = eP_Problem.name.format(m=m, k=k, L0=L0, g=g)

# CVode Solver
eP_Solver = asol.CVode(eP_Problem)
eP_Solver.reset() # Why is this needed here?
t_sol, x_sol = eP_Solver.simulate(4, 1000) # simulate(tf, ncp)


