from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import assimulo.problem as apro
import matplotlib.pyplot as mpl
import assimulo.solvers as asol
import NewmarkBetaSolverFelix as solver
import Explicit_Problem_2nd as EP2

# Paramters for the elastic pendulum
m = 1.0      # mass
k = 10  # spring constant
L0 = 1.0     # natural length of the spring
g = 1.0     # acceleration due to gravity
simulation_time = 9.4
communication_points = 1000

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

EP2_Problem = EP2(eP_Problem, 2)

exp_sim = solver.NewmarkBetaSolver(EP2_Problem, 0.0, 0.25)
exp_sim.reset()
simulation_time = 5
communication_points = 50
t, y = exp_sim.simulate(simulation_time, communication_points)
