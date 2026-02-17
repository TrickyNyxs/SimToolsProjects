from assimulo.problem import Explicit_Problem
import numpy as np

class Explicit_Problem_2nd(Explicit_Problem):
    """
    Wrapper class for second-order ODEs that were transformed
    into first-order systems.

    Assumes state y = [q, v], where
      q' = v
      v' = a(q, v, t)
    """

    def __init__(self, first_order_problem, n): # Initialize the class
        """
        Parameters
        ----------
        first_order_problem : Explicit_Problem
            The original first-order problem.
        n : int
            Number of second-order degrees of freedom.
        """
        self.first_order_problem = first_order_problem
        self.n = n
        super().__init__(self.rhs, first_order_problem.y0, first_order_problem.t0)

    def split_state(self, y): 
        """Split state into position and velocity."""
        q = y[:self.n] # Position components
        v = y[self.n:] # Velocity components
        return q, v

    def acceleration(self, t, q, v):
        """
        Compute acceleration from the original first-order RHS.
        """
        y = np.concatenate((q, v))
        ydot = self.first_order_problem.rhs(t, y)
        return ydot[self.n:]

    def rhs(self, t, y):
        q, v = self.split_state(y)
        a = self.acceleration(t, q, v)
        return np.concatenate((v, a))

    def force(self, t, q, v, M, C, K):
        """
        Compute external force f(t) from M q'' + C q' + K q = f.
        """
        a = self.acceleration(t, q, v)
        return M @ a + C @ v + K @ q

    def force_from_state(self, t, y, M, C, K):
        """
        Convenience wrapper: compute f(t) given stacked state y=[q,v].
        """
        q, v = self.split_state(y)
        return self.force(t, q, v, M, C, K)
