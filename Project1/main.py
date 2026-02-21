from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import assimulo.problem as apro
import matplotlib.pyplot as mpl
import assimulo.solvers as asol

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

def task4():
    RTOL_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    RTOL_error_list = []
    ATOL_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    ATOL_error_list = []
    MAXORD_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
    MAXORD_error_list = []
    
    eP_Problem = apro.Explicit_Problem(lambda t, y: elastic_pendulum(t, y), t0=0, y0=initial_conditions)
    eP_Solver = asol.CVode(eP_Problem)
    eP_Solver.rtol = 1e-10 # usually the default is  1e-4, but we can set it higher for a more accurate reference solution
    eP_Solver.atol = 1e-10 # usually the default is  1e-4, but we can set it higher for a more accurate reference solution
    eP_Solver.maxord = 20  # usually the default is    5 , but we can set it higher for a more accurate reference solution
    t_sol, y_sol = eP_Solver.simulate(simulation_time, 10000)  # fine reference

    for RTOL in RTOL_values:
        # Reference solution with fine steps
        eP_Solver.rtol = RTOL
        eP_Solver.reset()
        t, y = eP_Solver.simulate(simulation_time, 10000)  # fine reference
        error = np.linalg.norm(y_sol - y, ord=np.inf)
        RTOL_error_list.append(error)

    for ATOL in ATOL_values:
        # Reference solution with fine steps
        eP_Solver.atol = ATOL
        eP_Solver.reset()
        t, y = eP_Solver.simulate(simulation_time, 10000)  # fine reference
        error = np.linalg.norm(y_sol - y, ord=np.inf)
        ATOL_error_list.append(error)
    
    for MAXORD in MAXORD_values:
        eP_Solver.maxord = MAXORD
        eP_Solver.reset()
        t, y = eP_Solver.simulate(simulation_time, 10000)  # fine reference
        error = np.linalg.norm(y_sol - y, ord=np.inf)
        MAXORD_error_list.append(error)
    

    plt.figure()

    plt.loglog(RTOL_values, RTOL_error_list, marker='o', label='RTOL')
    plt.loglog(ATOL_values, ATOL_error_list, marker='o', label='ATOL')

    plt.xlabel('Relative/Absolute Tolerance')
    plt.ylabel('Norm of Error')
    plt.title('Error vs Relative and Absolute Tolerance for Elastic Pendulum')
    plt.grid(True, which="both", ls="--")
    plt.legend()   # <- this shows which curve is which
    plt.show()

    plt.figure()
    plt.loglog(MAXORD_values, MAXORD_error_list, marker='o', label='MAXORD')
    plt.xlabel('MAXORD Value')
    plt.ylabel('Norm of Error')
    plt.title('Error vs MAXORD for Elastic Pendulum')
    plt.grid(True, which="both", ls="--")
    plt.legend()   # <- this shows which curve is which
    plt.show()

task4()

def main():
    # CVode Solver
    eP_Solver = asol.CVode(eP_Problem)
    eP_Solver.reset() # Why is this needed here?
    t_sol, x_sol = eP_Solver.simulate(simulation_time, 1000) # simulate(tf, ncp)
    eP_Solver.reset()
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

    import BDF4Solver
    exp_sim = BDF4Solver.BDF4(eP_Problem) #Create a BDF solver
    exp_sim.reset()
    print(exp_sim.maxsteps)
    t, y = exp_sim.simulate(simulation_time)
    # Plot Results
    plt.figure()

    plt.plot(t, y[:,0], label=r'$x(t)$')
    plt.plot(t, y[:,1], label=r'$y(t)$')
    plt.plot(t, y[:,2], label=r'$v_x(t)$')
    plt.plot(t, y[:,3], label=r'$v_y(t)$')

    plt.xlabel('t')
    plt.ylabel('state value')
    plt.legend()
    plt.title("BDF4: " + eP_Problem.name)
    plt.grid(True)
    plt.show()

    #BDF2Solver
    import BDF2Solver 

    exp_sim = BDF2Solver.BDF_2(eP_Problem) #Create a BDF solver
    exp_sim.reset()
    t, y = exp_sim.simulate(simulation_time, communication_points)
    # Plot Results
    plt.figure()

    plt.plot(t, y[:,0], label=r'$x(t)$')
    plt.plot(t, y[:,1], label=r'$y(t)$')
    plt.plot(t, y[:,2], label=r'$v_x(t)$')
    plt.plot(t, y[:,3], label=r'$v_y(t)$')

    plt.xlabel('t')
    plt.ylabel('state value')
    plt.legend()
    plt.title("BDF2: " + eP_Problem.name)
    plt.grid(True)
    plt.show()

    #EESolver

    import EESolver
    exp_sim = EESolver.Explicit_Euler(eP_Problem) #Create a BDF solver
    exp_sim.reset()
    t, y = exp_sim.simulate(simulation_time, communication_points)

    # Plot Results
    plt.figure()

    plt.plot(t, y[:,0], label=r'$x(t)$')
    plt.plot(t, y[:,1], label=r'$y(t)$')
    plt.plot(t, y[:,2], label=r'$v_x(t)$')
    plt.plot(t, y[:,3], label=r'$v_y(t)$')

    plt.xlabel('t')
    plt.ylabel('state value')
    plt.legend()
    plt.title("EE: " + eP_Problem.name)
    plt.grid(True)
    plt.show()

    # Accuracy Testing

    # Define Problem 
    initial_conditions = [0.0, 1.1, 0.1, 0.0]  # initial position (x,y) and velocity (vx, vy)
    k_list = [1000, 100, 50, 20, 10, 5, 2, 1, 0.5, 0.2, 0.1]
    EE_error_list = []
    BDF4_error_list = []
    BDF2_error_list = []
    simulation_time = 4.9

    for k in k_list:
        print(k)
        def elastic_pendulum(t,y):
            yvec = np.zeros_like(y)
            yvec[0] = y[2]
            yvec[1] = y[3]
            lam = k * (np.sqrt(y[0]**2 + y[1]**2) - 1) / np.sqrt(y[0]**2 + y[1]**2)
            yvec[2] = -1*y[0]*lam
            yvec[3] = -1*y[1]*lam - 1
            return yvec
        
        eP_kProblem = apro.Explicit_Problem(elastic_pendulum, t0 = 0, y0 = initial_conditions)
        
        eP_Solver = asol.CVode(eP_kProblem)
        eP_Solver.reset() # Why is this needed here?
        t_sol, y_sol = eP_Solver.simulate(simulation_time, 491) # simulate(tf, ncp)

        # EE Solver
        exp_sim = EESolver.Explicit_Euler(eP_kProblem) #Create a BDF solver
        exp_sim.reset()
        t, y = exp_sim.simulate(simulation_time, 1000)
        error = np.linalg.norm(y_sol - y, ord=np.inf)
        EE_error_list.append(error)

        # BDF4 Solver
        exp_sim = BDF4Solver.BDF4(eP_kProblem) #Create a BDF solver
        exp_sim.reset()
        t, y = exp_sim.simulate(simulation_time, 1000)
        error = np.linalg.norm(y_sol - y, ord=np.inf)
        BDF4_error_list.append(error)

        # BDF2 Solver
        exp_sim = BDF2Solver.BDF_2(eP_kProblem) #Create a BDF solver
        exp_sim.reset()
        t, y = exp_sim.simulate(simulation_time, 1000)
        error = np.linalg.norm(y_sol - y, ord=np.inf)
        BDF2_error_list.append(error)
        


    plt.figure()

    plt.loglog(k_list, EE_error_list, marker='o', label='Explicit Euler')
    plt.loglog(k_list, BDF4_error_list, marker='x', label='BDF4')
    plt.loglog(k_list, BDF2_error_list, marker='s', label='BDF2')

    plt.xlabel('Spring Constant k')
    plt.ylabel('Norm of Error')
    plt.title('Error vs Spring Constant for Elastic Pendulum')
    plt.grid(True, which="both", ls="--")
    plt.legend()   # <- this shows which curve is which
    plt.show()
    

    # Stability Testing
    import numpy as np
    import matplotlib.pyplot as plt
    from math import ceil

    k_list = [100, 10,1]
    h_list = [0.1, 0.01, 0.005]
    EE_error_list = []
    BDF4_error_list = []
    BDF2_error_list = []
    simulation_time = 4.9

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
    import numpy as np
    import matplotlib.pyplot as plt
    from math import ceil
    def k_dependent_pendulum(t, y, k):
            yvec = np.zeros_like(y)
            yvec[0] = y[2]
            yvec[1] = y[3]
            r = np.sqrt(y[0]**2 + y[1]**2)
            lam = k * (r - 1) / r
            yvec[2] = -y[0] * lam
            yvec[3] = -y[1] * lam - 1
            return yvec


    def accuracy_testing():
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
        return

    def stability_test():
        EE_error_list = []
        BDF4_error_list = []
        BDF2_error_list = []

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

        # --------------------------------------------------
        # Plot all h values in the SAME figure
        # --------------------------------------------------

        plt.figure()

        markers = {
            "EE": "o",
            "BDF4": "x",
            "BDF2": "s"
        }

        linestyles = ["-", "--", ":"]  # one per h

        for i, h in enumerate(h_list):
            ls = linestyles[i % len(linestyles)]

            plt.loglog(
                k_list,
                [EE_error_list[j][i] for j in range(len(k_list))],
                marker=markers["EE"],
                linestyle=ls,
                label=f"EE (h = {h})"
            )

            plt.loglog(
                k_list,
                [BDF4_error_list[j][i] for j in range(len(k_list))],
                marker=markers["BDF4"],
                linestyle=ls,
                label=f"BDF4 (h = {h})"
            )

            plt.loglog(
                k_list,
                [BDF2_error_list[j][i] for j in range(len(k_list))],
                marker=markers["BDF2"],
                linestyle=ls,
                label=f"BDF2 (h = {h})"
            )

        plt.xlabel("Spring Constant k")
        plt.ylabel("Infinity Norm of Error")
        plt.title("Elastic Pendulum Error vs k (all h)")
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.show()
        return



