from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import ID_PY_OK, NORMAL
import inspect
import scipy.linalg as SL
import scipy.sparse as sps
import numpy as np
import scipy.linalg as SL
import scipy.sparse as sps
import scipy.sparse.linalg as spsl

class ExplicitNewmarkBetaSolver(Explicit_ODE):
    
    #Define variables
    tol=1.e-8     
    maxit=100
    
    
    def __init__(self, problem, M, C, K, f): #Initialize the class
        Explicit_ODE.__init__(self, problem) #Calls the base class

        self.beta = 0
        self.gamma = 0.25
        self.K = K
        self.C = C
        self.M = M
        self.f = f

        if not (0 <= self.beta <= 0.5):
            raise ValueError("beta must be in [0, 1/2]")
        if not (0 <= self.gamma <= 1):
            raise ValueError("gamma must be in [0, 1]")
        
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
        u_old, v_old = self.problem.split_state(y)
        u_old = u_old.reshape(-1,1)
        v_old = v_old.reshape(-1,1)


        # get a from (7)
        a_old = self.step_7(self.M, self.f(t, u_old), self.C, self.K(u_old), u_old, v_old)
     

        for _ in range(self.maxsteps):
            if t_it >= tf:
                break
            self.statistics["nsteps"] += 1

            u, v, a = self.step_Newmark([u_old, v_old, a_old], t)
            t_it += h

            # return a 1-D state vector [u; v] (Assimulo expects 1-D arrays)
            y = np.concatenate((u.flatten(), v.flatten()))
            tres.append(t_it)
            yres.append(y.copy())

            h = min(self.h, abs(tf - t_it))
        else:
            raise Exception("Final time not reached within maximum number of steps")

        return ID_PY_OK, tres, yres
        
        
    def step_Newmark(self,Y, t):
        """
        Newmark stepper for second-order ODEs.
        """

        self.statistics["nfcns"] += 1
        
        # evaluate force vector at current time and displacement
        ft = self.f(t, Y[0])

        if self.beta == 0: 
            # Explicit case (7) = (10), (8), (9)
            u_new = self.step_8(Y[0], Y[1], Y[2])
            a_new = self.step_7(self.M, ft, self.C, self.K(Y[0]), Y[0], Y[1])
            v_new = self.step_9(Y[1], Y[2], a_new)
        else: # Implicit
            u_new = self.step_7quote(self.M, ft, self.C, self.K, Y[0], Y[1], Y[2])
            v_new = self.step_6quote(self, Y[0], u_new, Y[1], Y[2])
            a_new = self.step_5quote(self, Y[0], u_new, v_new, Y[2])
        
        return [u_new, v_new, a_new]
        
        
    def step_7(self, M, ft, C, K, u, v): #Initializer to get u_0''
        return SL.solve(M, ft - C @ v - K @ u)
    
    def step_8(self , u_old, v_old, a_old): #Initializer to get u_0''
        return u_old + v_old*self.h + 0.5*self.h**2*a_old

    def step_9(self, v_old, a_old, a_new): #Initializer to get u_0''
        return v_old + (1 - self.gamma)*self.h*a_old + self.gamma*self.h*a_new

    def step_7quote(self, M, ft, C, K, u, v, a): # returns u_{n+1}
        K_eff = M/self.beta/self.h**2 + C*self.gamma/self.beta/self.h + K(u)
        rhs = ft + M @ (1/self.beta/self.h**2 * u + 1/self.beta/self.h * v + (1/2/self.beta - 1) * a) + C @ (self.gamma/self.beta/self.h * u + (self.gamma/self.beta - 1) * v + self.h/2*(self.gamma/self.beta - 2) * a)
        return SL.solve(K_eff, rhs)
    
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
