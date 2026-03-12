from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import *
import numpy as np
import matplotlib.pyplot as mpl
import scipy.linalg as SL
from assimulo.solvers import CVode
import assimulo.problem as apro

class BDF4(Explicit_ODE):
    
    #Define variables
    tol=1.e-8     
    maxit=100     
    
    def __init__(self, problem): #Initialize the class
        Explicit_ODE.__init__(self, problem) #Calls the base class
        
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
    
    def jacobian_fd(self, t, y):
        """
        Compute the Jacobian using finite differences.
        Works for any function without requiring analytical derivatives.
        
        Uses an adaptive epsilon based on the magnitude of y to avoid
        both truncation and round-off errors.
        """
        n = len(y)
        J = np.zeros((n, n))
        
        # Evaluate RHS at base point
        f = self.problem.rhs(t, y)
        
        # Finite difference for each column
        for j in range(n):
            # Adaptive epsilon: sqrt(machine eps) * max(|y_j|, 1)
            # This balances truncation error and round-off error
            eps = np.sqrt(np.finfo(float).eps) * max(abs(y[j]), 1.0)
            
            y_perturbed = y.copy()
            y_perturbed[j] += eps
            f_perturbed = self.problem.rhs(t, y_perturbed)
            J[:, j] = (f_perturbed - f) / eps
        
        return J
    
    def integrate(self, t, y, tf, opts):
        """
        _integrates (t,y) values until t > tf
        """
        h = self.options["h"]
        h = min(h, abs(tf-t))
        
        #Lists for storing the result
        tres = []
        yres = []
        
        # Store current and historical states
        # These will be properly populated as we take steps
        t_current = t
        y_current = y.copy()
        
        # History arrays (will be filled during startup)
        t_hist = [t]  # t_hist[0] = t_n, [-1] = t_{n-1}, [-2] = t_{n-2}, etc.
        y_hist = [y.copy()]
        
        for i in range(self.maxsteps):
            if t_current >= tf:
                break
            self.statistics["nsteps"] += 1
            
            # Choose method based on available history
            if i == 0:  # initial step - use Explicit Euler
                t_np1, y_np1 = self.step_EE(t_current, y_current, h)
                
            elif i == 1:  # one history point available - use BDF2
                t_np1, y_np1 = self.step_BDF2(
                    [t_hist[-1], t_hist[-2]], 
                    [y_hist[-1], y_hist[-2]], 
                    h
                )
                
            elif i == 2:  # two history points available - use BDF3
                t_np1, y_np1 = self.step_BDF3(
                    [t_hist[-1], t_hist[-2], t_hist[-3]], 
                    [y_hist[-1], y_hist[-2], y_hist[-3]], 
                    h
                )
                
            else:  # three or more history points - use BDF4
                t_np1, y_np1 = self.step_BDF4(
                    [t_hist[-1], t_hist[-2], t_hist[-3], t_hist[-4]], 
                    [y_hist[-1], y_hist[-2], y_hist[-3], y_hist[-4]], 
                    h
                )
            
            # Update current state
            t_current = t_np1
            y_current = y_np1.copy()
            
            # Add to history (most recent at end)
            t_hist.append(t_current)
            y_hist.append(y_current)
            
            # Store results
            tres.append(t_current)
            yres.append(y_current)
        
            h = min(self.h, np.abs(tf - t_current))
        else:
            raise Exception('Final time not reached within maximum number of steps')
        
        return ID_PY_OK, tres, yres
    
    def step_EE(self, t, y, h):
        """
        This calculates the next step in the integration with explicit Euler.
        """
        self.statistics["nfcns"] += 1
        
        f = self.problem.rhs
        return t + h, y + h*f(t, y) 
        
    def step_BDF2(self,T,Y, h):
        """
        BDF-2 with Newton's method
        
        alpha_0*y_np1+alpha_1*y_n+alpha_2*y_nm1=h f(t_np1,y_np1)
        alpha=[3/2,-2,1/2]
        """
        alpha=[3./2.,-2.,1./2]
        f=self.problem.rhs
        
        t_n,t_nm1=T
        y_n,y_nm1=Y
        # predictor
        t_np1=t_n+h
        y_np1_i=y_n   # zero order predictor
        
        # Newton iteration
        for i in range(self.maxit):
            self.statistics["nfcns"] += 1
            
            # Evaluate function at current iterate
            f_val = f(t_np1, y_np1_i)
            
            # Define the residual
            G = alpha[0]*y_np1_i - h*f_val + alpha[1]*y_n + alpha[2]*y_nm1
            
            res_norm = SL.norm(G)
            # Check convergence before computing expensive Jacobian
            if res_norm < self.tol:
                return t_np1, y_np1_i
            
            # Define the Jacobian of the residual
            J = alpha[0]*np.eye(len(y_n)) - h*self.jacobian_fd(t_np1, y_np1_i)
        
            delta_y = SL.solve(J, -G)
            y_np1_i = y_np1_i + delta_y
        else:
            raise Exception('Corrector could not converge within %d iterations (res_norm=%e)' % (i, res_norm))
    
    def step_BDF3(self,T,Y, h):
        """BDF-3 with Newton's method"""
        alpha = [11./6., -3., 3./2., -1./3.]
        
        t_n, t_nm1, t_nm2 = T
        y_n, y_nm1, y_nm2 = Y
        # predictor
        t_np1 = t_n + h
        y_np1_i = y_n  # zero order predictor
        
        # Newton iteration
        for i in range(self.maxit):
            self.statistics["nfcns"] += 1
            
            # Evaluate function at current iterate
            f_val = self.problem.rhs(t_np1, y_np1_i)
            
            # Define the residual
            G = alpha[0]*y_np1_i - h*f_val + alpha[1]*y_n + alpha[2]*y_nm1 + alpha[3]*y_nm2
            
            # Check convergence before computing expensive Jacobian
            if SL.norm(G) < self.tol:
                return t_np1, y_np1_i
            
            # Define the Jacobian of the residual
            J = alpha[0]*np.eye(len(y_n)) - h*self.jacobian_fd(t_np1, y_np1_i)
            
            delta_y = SL.solve(J, -G)
            y_np1_i = y_np1_i + delta_y
        else:
            raise Exception('Corrector could not converge within %d iterations' % i)
        
    def step_BDF4(self,T,Y, h):
        """BDF-4 with Newton's method"""
        alpha = [25./12., -4., 3., -4./3., 1./4.]
        
        t_n, t_nm1, t_nm2, t_nm3 = T
        y_n, y_nm1, y_nm2, y_nm3 = Y
        # predictor
        t_np1 = t_n + h
        y_np1_i = y_n  # zero order predictor
        
        # Newton iteration
        for i in range(self.maxit):
            self.statistics["nfcns"] += 1
            
            # Evaluate function at current iterate
            f_val = self.problem.rhs(t_np1, y_np1_i)
            
            # Define the residual
            G = alpha[0]*y_np1_i - h*f_val + alpha[1]*y_n + alpha[2]*y_nm1 + alpha[3]*y_nm2 + alpha[4]*y_nm3
            
            res_norm = SL.norm(G)
            # Check convergence before computing expensive Jacobian
            if res_norm < self.tol:
                return t_np1, y_np1_i
            
            # Define the Jacobian of the residual
            J = alpha[0]*np.eye(len(y_n)) - h*self.jacobian_fd(t_np1, y_np1_i)
            
            delta_y = SL.solve(J, -G)
            y_np1_i = y_np1_i + delta_y
        else:
            raise Exception('Corrector could not converge within %d iterations (res_norm=%e)' % (i, res_norm))
    
    def print_statistics(self, verbose=NORMAL):
        self.log_message('Final Run Statistics            : {name} \n'.format(name=self.problem.name),        verbose)
        self.log_message(' Step-length                    : {stepsize} '.format(stepsize=self.options["h"]), verbose)
        self.log_message(' Number of Steps                : '+str(self.statistics["nsteps"]),          verbose)               
        self.log_message(' Number of Function Evaluations : '+str(self.statistics["nfcns"]),         verbose)
            
        self.log_message('\nSolver options:\n',                                    verbose)
        self.log_message(' Solver            : BDF4',                     verbose)
        self.log_message(' Solver type       : Fixed step\n',                      verbose)
