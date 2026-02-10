import numpy as np
from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import *
import assimulo.problem as apro

class Second_Order(Explicit_ODE):

    tol = 1e-8
    maxit = 50

    def __init__(self, problem, beta, gamma):
        self.problem = problem

        if not (0 <= beta <= 0.5):
            raise ValueError("beta must be in [0, 1/2]")
        if not (0 <= gamma <= 1):
            raise ValueError("gamma must be in [0, 1]")

        self.beta = beta
        self.gamma = gamma
        self.dim = 2  # Dimension of the system
        Explicit_ODE.__init__(self, problem) #Calls the base class
        
        # Solver options
        self.options["h"] = 0.01
        self.maxsteps = 1000000

        # Statistics
        self.statistics["nsteps"] = 0
        self.statistics["nfcns"] = 0

    # -----------------------------------------------------
    # Required dummy RHS (not used)
    # -----------------------------------------------------
    def rhs(self, t, y):
        raise NotImplementedError("Second-order solver uses custom stepping.")

    # -----------------------------------------------------
    # Step-size property
    # -----------------------------------------------------
    def _set_h(self, h):
        self.options["h"] = float(h)

    def _get_h(self):
        return self.options["h"]

    h = property(_get_h, _set_h)

    # -----------------------------------------------------
    # Time integration loop
    # -----------------------------------------------------
    def integrate(self, t, y, tf, opts):

        h = min(self.h, abs(tf - t))

        tres = []
        yres = []
        
        print("shape y ", y.shape)
        print("y: ", y)
        # Known quantities
        """ 
        y0 =
        ydot0 = 
        t0 = t
        """

        q = y[:self.dim]
        v = y[self.dim:]
        a = self.problem.acceleration(t, q, v)

        for _ in range(self.maxsteps):
            if t >= tf:
                break

            self.statistics["nsteps"] += 1

            t, q, v, a = self.step_newmark(t, q, v, a, h)

            y = np.hstack((q, v))
            tres.append(t)
            yres.append(y.copy())

            h = min(self.h, abs(tf - t))

        else:
            raise Exception("Final time not reached")

        return ID_PY_OK, tres, yres
    

    def step_newmark(self, t, q, v, a, h):
        """
        Perform a single Newmark time step.
        """
        self.statistics["nfcns"] += 1

        # Predict displacement and velocity
        q_pred = q + h * v + (0.5 - self.beta) * h**2 * a
        v_pred = v + (1 - self.gamma) * h * a

        # Initialize acceleration
        a_new = a.copy()

        for _ in range(self.maxit):
            # Compute residual
            res = self.problem.rhs(t + h, q_pred) - a_new

            if np.linalg.norm(res) < self.tol:
                break

            # Update acceleration
            a_new += res

        else:
            raise Exception("Newton-Raphson did not converge")

        # Correct displacement and velocity
        q_new = q_pred + self.beta * h**2 * a_new
        v_new = v_pred + self.gamma * h * a_new

        return t + h, q_new, v_new, a_new