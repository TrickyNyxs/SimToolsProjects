import numpy as np
import matplotlib.pyplot as plt
import assimulo.problem as apro
import matplotlib.pyplot as mpl
import assimulo.solvers as asol
from Explicit_Problem_2nd import Explicit_Problem_2nd as EP2
from ExplicitNBSolverDante import ExplicitNewmarkBetaSolver as ExplicitNBSolverD

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
EP2_Problem = EP2(eP_Problem, 2)
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
beta, gamma = 0, 0.25
NBSolverDante = ExplicitNBSolverD(EP2_Problem, M, C, K, f, beta, gamma)
t, y = NBSolverDante.simulate(simulation_time,1000)
NBSolverDante.reset()

plt.figure()

plt.plot(t, y[:,0], label=r'$x(t)$')
plt.plot(t, y[:,1], label=r'$y(t)$')
plt.plot(t, y[:,2], label=r'$v_x(t)$')
plt.plot(t, y[:,3], label=r'$v_y(t)$')

plt.xlabel('t')
plt.ylabel('state value')
plt.legend()
plt.title(eP_Problem.name)
plt.grid(True)
plt.show()