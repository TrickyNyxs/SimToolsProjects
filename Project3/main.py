import numpy as np
import matplotlib.pyplot as plt
import assimulo.problem as apro
import matplotlib.pyplot as mpl
import assimulo.solvers as asol

eta = 10
v = 0.1

def y(x): # manufactured solution that is 0 at the boundary
    return np.sin(np.pi*x) 

def y_hat(x):
    return (1 + v*(eta + np.pi**2)**2) * y(x)

def u(x): # calculated from the manufactured solution
    return (np.pi**2 + eta)*y(x)

def p(x): # calculated from the manufactured solution
    return v*(np.pi**2 + eta)*y(x)

def plot_true_solution():
    h = 0.01
    x = np.arange(0, 1+h, h)
    plt.plot(x, y(x), label='Manufactured Solution')
    plt.plot(x, p(x), label='Calculated p(x)')
    plt.xlabel('x')
    plt.ylabel('Value')
    plt.title('Manufactured Solution and Calculated u(x) and p(x)')
    plt.legend()
    plt.grid(True)
    plt.show()
    return y(x)

eta = 10.0
v   = 0.1
h   = 0.01
x   = np.arange(0, 1+h, h)

# Build 1D Laplacian (interior points only)
n = len(x) - 2
L = np.zeros((n, n))
np.fill_diagonal(L, -2)
np.fill_diagonal(L[1:], 1)
np.fill_diagonal(L[:, 1:], 1)
L = L / h**2


def solve_optimality_system(L, y_hat_vec, eta, nu):
    n = L.shape[0]
    I = np.eye(n)
    A = eta*I - L
    M = nu * (A @ A) + I
    y = np.linalg.solve(M, y_hat_vec)
    p = nu * (A @ y)
    return y, p

yh_int = y_hat(x[1:-1])
y_int, p_int = solve_optimality_system(L, yh_int, eta, v)

# add Dirichlet boundaries
y_sol = np.concatenate(([0.0], y_int, [0.0]))
p_sol = np.concatenate(([0.0], p_int, [0.0]))

print("max |y_sol| =", np.max(np.abs(y_sol)))
print("max |p_sol| =", np.max(np.abs(p_sol)))

plt.plot(x, y(x), label="y_true")
plt.plot(x, y_sol, "--", label="y_num")
plt.legend()
plt.grid(True)
plt.show()


# Refinement analysis

h_list = [1/20, 1/40, 1/80, 1/160, 1/320] # instead of increaing nbr of unkowns by 1/h we increase it by 

y_errors = []
p_errors = []
for h in h_list:
    x = np.arange(0, 1+h, h)
    n = len(x) - 2
    L = np.zeros((n, n))
    np.fill_diagonal(L, -2)
    np.fill_diagonal(L[1:], 1)
    np.fill_diagonal(L[:, 1:], 1)
    L = L / h**2
    
    yh_int = y_hat(x[1:-1])
    y_int, p_int = solve_optimality_system(L, yh_int, eta, v)
    
    y_sol = np.concatenate(([0.0], y_int, [0.0]))
    p_sol = np.concatenate(([0.0], p_int, [0.0]))
    
    y_errors.append(np.linalg.norm(y_sol - y(x), ord=np.inf))
    p_errors.append(np.linalg.norm(p_sol - p(x), ord=np.inf))

plt.figure() 
plt.plot(h_list, y_errors, "o-", label="y_error") 
plt.plot(h_list, p_errors, "s-", label="p_error") 
plt.xlabel("Grid Spacing (h)") 
plt.ylabel("Error") 
plt.title("Refinement Analysis") 
plt.legend() 
plt.grid(True)
plt.show()