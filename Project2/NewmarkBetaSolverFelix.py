from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import *
import numpy as np
import matplotlib.pyplot as mpl
import scipy.linalg as SL
from assimulo.solvers import CVode
import assimulo.problem as apro

class NewmarkBetaSolver(Explicit_ODE):
    
    #Define variables
    tol=1.e-8     
    maxit=100
    beta = 0.25
    gamma = 0.5
    
    
    def __init__(self, problem, b, g): #Initialize the class
        Explicit_ODE.__init__(self, problem) #Calls the base class
        self.beta = b
        self.gamma = g
        
        
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
        _integrates (t,y) values until t > tf
        """
        h = self.options["h"]
        h = min(h, abs(tf-t))
        
        #Lists for storing the result
        tres = []
        yres = []
        
        # Initialize historical states to current state (will be overwritten on first iterations)
        t_nm1,t_nm2,t_nm3= t, t, t
        y_nm1,y_nm2,y_nm3= y.copy(), y.copy(), y.copy()
        
        for i in range(self.maxsteps):
            if t >= tf:
                break
            self.statistics["nsteps"] += 1
            
            if i==0:  # initial step
                t_np1,y_np1 = self.step_EE(t,y, h)
            elif i==1:
                t_np1, y_np1 = self.step_BDF2([t, t_nm1], [y, y_nm1], h)
            elif i==2: 
                t_np1, y_np1 = self.step_BDF3([t, t_nm1, t_nm2], [y, y_nm1, y_nm2], h)
            else:   
                t_np1, y_np1 = self.step_BDF4([t,t_nm1, t_nm2, t_nm3], [y,y_nm1, y_nm2, y_nm3], h)
            t,t_nm1,t_nm2,t_nm3=t_np1,t,t_nm1,t_nm2
            y,y_nm1,y_nm2,y_nm3=y_np1,y,y_nm1,y_nm2
            
            tres.append(t)
            yres.append(y.copy())
        
            h=min(self.h,np.abs(tf-t))
        else:
            raise Exception('Final time not reached within maximum number of steps')
        
        return ID_PY_OK, tres, yres
        
    def step_Newmark(self,T,Y, h):
        """
        Newmark stepper for second-order ODEs.
        """

        f=self.problem.rhs
        a_n+1
        
        
        for i in range(self.maxit):
            self.statistics["nfcns"] += 1
            
            
        
            delta_y = SL.solve(J, -G)
            y_np1_ip1= y_np1_i + delta_y
            
            if SL.norm(delta_y) < self.tol:
                return t_np1,y_np1_ip1
            y_np1_i=y_np1_ip1
        else:
            raise Exception('Corrector could not converge within %d iterations'%i)
        
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
