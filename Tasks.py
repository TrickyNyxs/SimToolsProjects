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

# Plot Results
plt.figure()

plt.plot(t_sol, x_sol[:,0], label=r'$x(t)$')
plt.plot(t_sol, x_sol[:,1], label=r'$y(t)$')
plt.plot(t_sol, x_sol[:,2], label=r'$v_x(t)$')
plt.plot(t_sol, x_sol[:,3], label=r'$v_y(t)$')

plt.xlabel('t')
plt.ylabel('state value')
plt.legend()
plt.title(eP_Problem.name)
plt.grid(True)
plt.show()

# Our Solution with BDF4
exp_sim = BDF4Solver.BDF4(eP_Problem) #Create a BDF solver
exp_sim.reset()
t, y = exp_sim.simulate(4)
exp_sim.plot()
mpl.show()


# Provided BDF2 Solution
exp_sim3 = BDF2Solver.BDF_2(eP_Problem) #Create a BDF solver
exp_sim3.reset()
t, y = exp_sim3.simulate(4)
exp_sim3.plot()
mpl.show()

