from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import ID_PY_OK, NORMAL
import inspect
import numpy as np
import scipy.linalg as SL
import scipy.sparse as sps
import scipy.sparse.linalg as spsl

class NewmarkBetaSolver(Explicit_ODE):
    
    #Define variables
    tol=1.e-8     
    maxit=100
    beta = 0.25
    gamma = 0.5
    
    
    def __init__(self, problem, b, g, K, C, M): #Initialize the class
        Explicit_ODE.__init__(self, problem) #Calls the base class

        if not (0 <= b <= 0.5):
            raise ValueError("beta must be in [0, 1/2]")
        if not (0 <= g <= 1):
            raise ValueError("gamma must be in [0, 1]")

        self.beta = b
        self.gamma = g
        self.K = K
        self.C = C
        self.M = M
        
        #Solver options
        self.options["h"] = 0.01
        self.maxsteps = 1000000  # Instance variable, not class variable
        
        #Statistics
        self.statistics["nsteps"] = 0
        self.statistics["nfcns"] = 0
    
    def _set_h(self,h):
            self.options["h"] = float(h)

    def _get_h(self):
        return self.options["h"]
        
    h=property(_get_h,_set_h) #Allows to get and set h as a property
    
    
    def integrate(self, t, y, tf, opts):
        """
        Integrates (t,y) until t >= tf.
        Expected state: y = [u, v] stacked.
        """
        h = min(self.h, abs(tf - t))

        # Lists for storing the result
        tres = []
        yres = []

        t_it = t
        u_old = y[0]
        v_old = y[1]
        a_old = self.step_7(self.M, f, self.C, self.K, u_old, v_old)
        

        for _ in range(self.maxsteps):
            if t_it >= tf:
                break
            self.statistics["nsteps"] += 1

            u, v, a = self.step_Newmark(self, [u_old, v_old, a_old])
            t_it += h

            y = np.hstack((u, v, a))
            tres.append(t_it)
            yres.append(y.copy())

            h = min(self.h, abs(tf - t_it))
        else:
            raise Exception("Final time not reached within maximum number of steps")

        return ID_PY_OK, tres, yres
        
        
    def step_Newmark(self,Y):
        """
        Newmark stepper for second-order ODEs.
        """

        self.statistics["nfcns"] += 1
        
        f=self.problem.rhs
        u_new = self.step_7quote(self, self.M, f, self.C, self.K, Y[0], Y[1], Y[2])
        v_new = self.step_6quote(self, Y[0], u_new, Y[1], Y[2])
        a_new = self.step_5quote(self, Y[0], u_new, v_new, Y[2])
        
        return [u_new, v_new, a_new]
        
        
    def step_7(M, f, C, K, u, v): #Initializer to get u_0''
        return SL.linalg.solve(M, f - C @ v - K @ u)
    
    def step_7quote(self, M, f, C, K, u, v, a): # returns u_{n+1}
        Eff = M/self.beta/self.h**2 + C*self.gamma/self.beta/self.h + K
        rhs = f + M @ (1/self.beta/self.h**2 * u + 1/self.beta/self.h * v + (1/2/self.beta - 1) * a) + C @ (self.gamma/self.beta/self.h * u + (self.gamma/self.beta - 1) * v + self.h/2*(self.gamma/self.beta - 2) * a)
        return SL.linalg.solve(Eff, rhs)
    
    def step_6quote(self, u_old, u, v_old, a_old): # returns u_{n+1}'
        return self.gamma/self.beta/self.h*(u-u_old) + (1 -self.gamma/self.beta)*v_old + self.h*(1 - self.gamma/self.beta/2)*a_old
    
    def step_5quote(self, u_old, u, v_old, a_old): # returns u_{n+1}''
        return 1/self.beta/self.h**2*(u-u_old) - 1/self.beta/self.h*v_old - (1/2/self.beta - 1)*a_old
    
    
    def print_statistics(self, verbose=NORMAL):
        self.log_message('Final Run Statistics            : {name} \n'.format(name=self.problem.name),        verbose)
        self.log_message(' Step-length                    : {stepsize} '.format(stepsize=self.options["h"]), verbose)
        self.log_message(' Number of Steps                : '+str(self.statistics["nsteps"]),          verbose)               
        self.log_message(' Number of Function Evaluations : '+str(self.statistics["nfcns"]),         verbose)
            
        self.log_message('\nSolver options:\n',                                    verbose)
        self.log_message(' Solver            : BDF4',                     verbose)
        self.log_message(' Solver type       : Fixed step\n',                      verbose)
