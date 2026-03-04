import numpy as np
import matplotlib.pyplot as plt
import assimulo.problem as apro
import matplotlib.pyplot as mpl
import assimulo.solvers as asol
from Explicit_Problem_2nd import Explicit_Problem_2nd as EP2
from NewmarkBetaSolverDante import NewmarkBetaSolver as NBSolver
from BDF4Solver import BDF4 as BDF4Solver
# Define first order problem
k = 10

def elastic_pendulum(t,y):
    yvec = np.zeros_like(y)
    yvec[0] = y[2]
    yvec[1] = y[3]
    lam = k * (np.sqrt(y[0]**2 + y[1]**2) - 1) / np.sqrt(y[0]**2 + y[1]**2)
    yvec[2] = -1*y[0]*lam
    yvec[3] = -1*y[1]*lam - 1
    return yvec


initial_conditions = [0.0, 1.1, 0.1, 0.0]  # initial position (x,y) and velocity (vx, vy)
eP_Problem = apro.Explicit_Problem(elastic_pendulum, t0 = 0, y0 = initial_conditions)
EP2_Problem = EP2(eP_Problem, 2) # transform 1 to 2nd order ODE
EP2_Problem.name = 'Elastic Pendulum Second Order Problem'
EP2_Problem.rhs(0, initial_conditions)



# Construct Stiffness Matrix, Dampening Matrix and Mass Matrix for pendulum
M = np.eye(2)
C = np.zeros((2, 2))
f = np.array([0.0, -1.0])

def K(y):
    return np.eye(2)* k * (np.sqrt(y[0]**2 + y[1]**2) - 1) / np.sqrt(y[0]**2 + y[1]**2)

def f(t,y):
    return np.array([0.0, -1.0]).reshape(-1,1)

# Solve and Simulate 
simulation_time = 20
beta = 0
gamma = 0.25
NBSolverDante = NBSolver(EP2_Problem, M, C, K, f, beta, gamma)
t_newmark, y_newmark = NBSolverDante.simulate(simulation_time, 1000)
NBSolverDante.reset()

plt.figure()

plt.plot(t_newmark, y_newmark[:,0], label=r'$x(t)$')
plt.plot(t_newmark, y_newmark[:,1], label=r'$y(t)$')
plt.plot(t_newmark, y_newmark[:,2], label=r'$v_x(t)$')
plt.plot(t_newmark, y_newmark[:,3], label=r'$v_y(t)$')

plt.xlabel('t')
plt.ylabel('state value')
plt.legend()
plt.title(eP_Problem.name)
plt.grid(True)
plt.show()

# BDF4Solver
exp_sim = BDF4Solver(eP_Problem) #Create a BDF solver (class imported as BDF4)
t_bdf4, x_bdf4 = exp_sim.simulate(simulation_time, 1000)
exp_sim.reset()
# Plot Results
plt.figure()

plt.plot(t_bdf4, x_bdf4[:,0], label=r'$x(t)$')
plt.plot(t_bdf4, x_bdf4[:,1], label=r'$y(t)$')
plt.plot(t_bdf4, x_bdf4[:,2], label=r'$v_x(t)$')
plt.plot(t_bdf4, x_bdf4[:,3], label=r'$v_y(t)$')

plt.xlabel('t')
plt.ylabel('state value')
plt.legend()
plt.title("BDF4: " + eP_Problem.name)
plt.grid(True)
plt.show()

# Differenve between the two solvers
plt.figure()

plt.plot(t_newmark, y_newmark[:,0] - x_bdf4[:,0],
         label=r'$x^{\mathrm{Newmark}} - x^{\mathrm{BDF4}}$')

plt.plot(t_newmark, y_newmark[:,1] - x_bdf4[:,1],
         label=r'$y^{\mathrm{Newmark}} - y^{\mathrm{BDF4}}$')

plt.plot(t_newmark, y_newmark[:,2] - x_bdf4[:,2],
         label=r'$v_x^{\mathrm{Newmark}} - v_x^{\mathrm{BDF4}}$')

plt.plot(t_newmark, y_newmark[:,3] - x_bdf4[:,3],
         label=r'$v_y^{\mathrm{Newmark}} - v_y^{\mathrm{BDF4}}$')

plt.xlabel('t')
plt.ylabel('difference in state value')
plt.legend()
plt.show()