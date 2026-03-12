import numpy as np
import matplotlib.pyplot as plt
import assimulo.problem as apro
import matplotlib.pyplot as mpl
import assimulo.solvers as asol
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Project1'))
from Explicit_Problem_2nd import Explicit_Problem_2nd as EP2
from NewmarkbetaSolverPendulum import NewmarkBetaSolver as NBSolver
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
    # Handle both (2,) and (2,1) shaped arrays
    y_flat = np.asarray(y).flatten()
    r = np.sqrt(y_flat[0]**2 + y_flat[1]**2)
    if r == 0:
        return np.zeros((2, 2))  # Handle singularity at origin
    return np.eye(2) * k * (r - 1) / r

def f(t,y):
    return np.array([0.0, -1.0]).reshape(-1,1)

# === Convergence Study: Variable Step Sizes ===
import assimulo.solvers as aso
from BDF4Solver import BDF4

simulation_time = 5.0
beta, gamma = 0, 0.5

# Generate fine reference solution (BDF4 with very small h)
print("\n--- Generating Reference Solution (BDF4, h=0.001) ---")
ref_bdf4 = BDF4(eP_Problem)
ref_bdf4._set_h(0.001)
t_ref, y_ref = ref_bdf4.simulate(simulation_time)
ref_bdf4.reset()

# Test multiple step sizes
h_values = [0.2, 0.1, 0.05, 0.02, 0.01]
results = {'h': [], 'Newmark_L2_error': [], 'BDF4_L2_error': []}

print("\n--- Convergence Study ---")
print(f"{'h':>10} {'Newmark L2 Error':>20} {'BDF4 L2 Error':>20}")
print("-" * 52)

for h in h_values:
    # Run Newmark-beta (explicit, beta=0)
    nmk = NBSolver(EP2_Problem, M, C, K, f, beta, gamma)
    nmk._set_h(h)
    t_nmk, y_nmk = nmk.simulate(simulation_time, int(simulation_time / h) + 100)
    nmk.reset()
    
    # Run BDF4
    bdf4_solver = BDF4(eP_Problem)
    bdf4_solver._set_h(h)
    t_bdf4, y_bdf4 = bdf4_solver.simulate(simulation_time)
    bdf4_solver.reset()
    
    # Interpolate reference to match time points (use nearest neighbor for simplicity)
    def nearest_ref_index(t):
        return np.argmin(np.abs(np.array(t_ref) - t))
    
    # Compute L2 error for Newmark
    y_ref_at_nmk = np.array([y_ref[nearest_ref_index(t)] for t in t_nmk])
    nmk_error = np.linalg.norm(y_nmk - y_ref_at_nmk) / np.sqrt(len(t_nmk))
    
    # Compute L2 error for BDF4
    y_ref_at_bdf4 = np.array([y_ref[nearest_ref_index(t)] for t in t_bdf4])
    bdf4_error = np.linalg.norm(y_bdf4 - y_ref_at_bdf4) / np.sqrt(len(t_bdf4))
    
    results['h'].append(h)
    results['Newmark_L2_error'].append(nmk_error)
    results['BDF4_L2_error'].append(bdf4_error)
    
    print(f"{h:10.4f} {nmk_error:20.4e} {bdf4_error:20.4e}")

# Plot convergence: log-log scale
plt.figure(figsize=(10, 6))
plt.loglog(results['h'], results['Newmark_L2_error'], 'o-', label='Newmark-β (explicit)', linewidth=2, markersize=8)
plt.loglog(results['h'], results['BDF4_L2_error'], 's-', label='BDF4 (implicit)', linewidth=2, markersize=8)
plt.xlabel('Step Size h', fontsize=12)
plt.ylabel('L2 Error vs Reference', fontsize=12)
plt.title('Convergence Study: Elastic Pendulum', fontsize=13)
plt.grid(True, which='both', alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('pendulum_errors_h.png', dpi=150, bbox_inches='tight')
print("Saved convergence plot to pendulum_errors_h.png")
plt.show()

# Single run with default h=0.01 for detailed comparison
print("\n--- Detailed Comparison (h=0.01) ---")
nmk_final = NBSolver(EP2_Problem, M, C, K, f, beta, gamma)
t_N, y_N = nmk_final.simulate(simulation_time, 1000)
nmk_final.reset()

bdf4_final = BDF4(eP_Problem)
t_bdf4, y_bdf4 = bdf4_final.simulate(simulation_time)
bdf4_final.reset()

# Error comparison
delta_x = np.abs(y_N[:,0] - y_bdf4[:,0])
delta_y = np.abs(y_N[:,1] - y_bdf4[:,1])
delta_vx = np.abs(y_N[:,2] - y_bdf4[:,2])
delta_vy = np.abs(y_N[:,3] - y_bdf4[:,3])

plt.figure(figsize=(10, 6))
plt.semilogy(t_N, delta_x, label=r'$\Delta x(t)$', linewidth=1.5)
plt.semilogy(t_N, delta_y, label=r'$\Delta y(t)$', linewidth=1.5)
plt.semilogy(t_N, delta_vx, label=r'$\Delta v_x(t)$', linewidth=1.5)
plt.semilogy(t_N, delta_vy, label=r'$\Delta v_y(t)$', linewidth=1.5)
plt.xlabel('t', fontsize=12)
plt.ylabel('Absolute Difference |Newmark - BDF4|', fontsize=12)
plt.title('Pointwise Error: Explicit Newmark-β vs BDF4 (h=0.01)', fontsize=13)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('pendulum_difference.png', dpi=150, bbox_inches='tight')
print("Saved difference plot to pendulum_difference.png")
plt.show()
plt.title("Absolute Difference between Newmark Beta and CVODE")
plt.grid(True)
plt.show()
