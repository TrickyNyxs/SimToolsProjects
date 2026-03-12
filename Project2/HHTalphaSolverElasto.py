from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import ID_PY_OK, NORMAL
import inspect
import numpy as np
import scipy.linalg as SL
import scipy.sparse as sps
import scipy.sparse.linalg as spsl

class HHTalphaSolverElasto(Explicit_ODE):
    
    #Define variables
    tol=1.e-8     
    maxit=100
    beta = 0.25
    gamma = 0.5
    
    
    def __init__(self, problem, a, K, C, M,f): #Initialize the class
        Explicit_ODE.__init__(self, problem) #Calls the base class

        if not (-1/3 <= a <= 0):
            raise ValueError("alpha must be in [-1/3, 0]")
        self.alpha = a
        self.beta = 0.25*(1 - self.alpha)**2
        self.gamma = 0.5 - self.alpha
        self.problem = problem
        self.K = K
        self.C = C
        self.M = M
        self.f = f
        
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
        a_old = self.step_7(self.M , self.f(t_it), self.C , self.K , u_old, v_old)
        u = np.copy(u_old)
        v = np.copy(v_old)
        a = np.copy(a_old)

        for _ in range(self.maxsteps):
            if t_it >= tf:
                break
            self.statistics["nsteps"] += 1
            
            t_it += h
            u, v, a = self.step_HHT([u_old, v_old, a_old], self.f(t_it), self.f(t_it-h))
            
            # Append Quantitites
            y = np.hstack((u, v))
            tres.append(t_it)
            yres.append(y.copy())

            #Update Quantities
            u_old, v_old, a_old = u, v, a

            h = min(self.h, abs(tf - t_it))
        else:
            raise Exception("Final time not reached within maximum number of steps")

        return ID_PY_OK, tres, yres
        
        
    def step_HHT(self, Y, f, fm1):
        """
        Newmark stepper for second-order ODEs.
        """

        self.statistics["nfcns"] += 1
        
        u_new = self.step_7quote(self.M, f, fm1, self.C, self.K, Y)
        v_new = self.step_6quote(Y[0], u_new, Y[1], Y[2])
        a_new = self.step_5quote(Y[0], u_new, Y[1], Y[2])
        
        return [u_new, v_new, a_new]
        
        
    def step_7(self, M, f, C, K, u, v): #Initializer to get u_0''
        rhs = f - C @ v - K @ u
        if sps.issparse(M):
            return spsl.spsolve(M, rhs)
        return SL.solve(M, rhs)
    
    def step_7quote(self, M, f, fm1,C, K, Y): # returns u_{n+1}
        Eff = M/self.beta/self.h**2 + (1+self.alpha)*C*self.gamma/self.beta/self.h + (1+self.alpha)*K
        rhs = (1+self.alpha)*f-self.alpha*fm1 + M @ (Y[0]/self.beta/self.h**2 + Y[1]/self.beta/self.h + (1/2/self.beta - 1) * Y[2]) + (1+self.alpha)*C @ (self.gamma/self.beta/self.h * Y[0] + (self.gamma/self.beta - 1) * Y[1] + self.h/2*(self.gamma/self.beta - 2) * Y[2]) + self.alpha*C @ Y[1] + self.alpha*K @ Y[0]
        if sps.issparse(Eff):
            return spsl.spsolve(Eff, rhs)
        return SL.solve(Eff, rhs)
    
    def step_6quote(self, u_old, u, v_old, a_old):
        return self.gamma/self.beta/self.h*(u-u_old) \
            + (1 - self.gamma/self.beta)*v_old \
            + self.h*(1 - self.gamma/self.beta/2)*a_old

    
    def step_5quote(self, u_old, u, v_old, a_old):
        return 1/self.beta/self.h**2*(u-u_old) \
            - 1/self.beta/self.h*v_old \
            - (1/2/self.beta - 1)*a_old

    
    
    def print_statistics(self, verbose=NORMAL):
        self.log_message('Final Run Statistics            : {name} \n'.format(name=self.problem.name),        verbose)
        self.log_message(' Step-length                    : {stepsize} '.format(stepsize=self.options["h"]), verbose)
        self.log_message(' Number of Steps                : '+str(self.statistics["nsteps"]),          verbose)               
        self.log_message(' Number of Function Evaluations : '+str(self.statistics["nfcns"]),         verbose)
            
        self.log_message('\nSolver options:\n',                                    verbose)
        self.log_message(' Solver            : HHT-alpha',                     verbose)
        self.log_message(' Solver type       : Fixed step\n',                      verbose)
