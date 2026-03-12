# -*- coding: utf-8 -*-
"""
@author: Peter Meisrimel, Robert Kloefkorn, Lund University
originally based on : https://comet-fenics.readthedocs.io/en/latest/demo/elastodynamics/demo_elastodynamics.py.html
"""

# Import of Solvers
import NewmarkbetaElasto as BetaSolver
import HHTalphaSolverElasto as HHTSolver
import Explicit_Problem_2nd as EP2
# Definition of variables

gamma = 0.6
beta = 0.25*(gamma + 0.5)**2
alpha = -0.3

import os
# ensure some compilation output for this example
os.environ['DUNE_LOG_LEVEL'] = 'info'
print("Using DUNE_LOG_LEVEL=",os.getenv('DUNE_LOG_LEVEL'))

import setuptools
import matplotlib.pyplot as pl
import numpy as np

from ufl import *
from dune.ufl import Constant, DirichletBC
from dune.grid import structuredGrid as leafGridView
from dune.fem.space import lagrange as lagrangeSpace
from dune.fem.space import dgonb as dgSpace

from dune.fem.operator import galerkin as galerkinOperator

import scipy.sparse as ssp
import scipy.sparse.linalg as ssl
pl.close('all')

class elastodynamic_beam:
    # Elastic parameters
    E  = 1000.0
    nu = 0.3
    mu = Constant(E / (2.0*(1.0 + nu)), name="mu")
    lmbda = Constant(E*nu / ((1.0 + nu)*(1.0 - 2.0*nu)), name="lmbda")

    # Mass density
    rho = Constant(1.0, name="rho")

    # Rayleigh damping coefficients
    eta_m = Constant(0.2, name="eta_m")
    eta_k = Constant(0.02, name="eta_k")

    def __init__(self, gridsize, T = 4.0, dimgrid=2):

        lower, upper, cells = [0., 0.], [1., 0.1], [12*gridsize, 2*gridsize]
        if dimgrid == 3:
            lower.append( 0 )
            upper.append( 0.04 )
            cells.append( gridsize )

        self.mesh = leafGridView( lower, upper, cells )
        dim = self.mesh.dimension

        # force only up to a certain time
        self.p0 = 1.
        self.cutoff_Tc = T/5
        # Define the loading as an expression
        self.p = self.p0/self.cutoff_Tc

        # Define function space for displacement (velocity and acceleration)
        self.V = lagrangeSpace( self.mesh, dimRange=dim, order=1)

        # UFL representative for x-coordinate
        x = SpatialCoordinate( self.V )

        self.bc = DirichletBC(self.V, as_vector(dim*[0]) , x[0]<1e-10)

        # Stress tensor
        def epsilon(r): # this is exactly dol.sym
            return 0.5*(nabla_grad(r) + nabla_grad(r).T)
        def sigma(r):
            return nabla_div(r)*Identity(dim) + 2.0 * self.mu * epsilon(r)

        # trial and test functions
        u = TrialFunction( self.V )
        v = TestFunction( self.V )

        # Mass form
        self.Mass_form = self.rho * inner( u, v ) * dx
        # Elastic stiffness form
        self.Stiffness_form = inner(sigma(u), epsilon(v))*dx
        # Rayleigh damping form
        self.Dampening_form = self.eta_m*self.Mass_form + self.eta_k*self.Stiffness_form

        self.M_FE = galerkinOperator([self.Mass_form,self.bc])      # mass matrix
        self.K_FE = galerkinOperator([self.Stiffness_form,self.bc]) # stiffness matrix
        self.D_FE = galerkinOperator([self.Dampening_form,self.bc]) # damping matrix

        # linear operator to assemble system matrices
        self.Mass_mat = self.M_FE.linear().as_numpy.tocsc()
        self.Stiffness_mat = self.K_FE.linear().as_numpy.tocsc()
        self.Dampening_mat = self.D_FE.linear().as_numpy.tocsc()

        # Work of external forces
        dimvec = dim*[0]
        pvec = dim*[0]
        pvec[ 1 ] = conditional(x[0] < 1 - 1e-10, 1, 0)*self.p*0.1
        pvec = as_vector(pvec)

        # right hand side
        self.External_forces_form = (inner( u, v) - inner(u,v))*dx + inner( v, pvec ) * ds

        self.F_ext = galerkinOperator(self.External_forces_form)

        self.ndofs = self.V.size
        self.F = np.zeros( self.V.size )
        self.fh = self.V.interpolate(dim*[0], name="fh")
        self.rh = self.V.interpolate(dim*[0], name="rh")

        self.F_ext( self.fh, self.rh )

        rh_np = self.rh.as_numpy
        self.F[:] = rh_np[:]

        print('degrees of freedom: ', self.ndofs)

    def res(self, t, y, yp, ypp):
        if t < self.cutoff_Tc:
            return self.Mass_mat@ypp * self.Stiffness_mat@yp + self.Dampening_mat@y - t*self.F
        else:
            return self.Mass_mat@ypp * self.Stiffness_mat@yp + self.Dampening_mat@y

    def rhs(self,t,y):
        Ft = t*self.F if t < self.cutoff_Tc else np.zeros(self.ndofs)
        return np.hstack((y[self.ndofs:],
                          ssl.spsolve(self.Mass_mat, -self.Stiffness_mat@y[:self.ndofs]
                                                     -self.Dampening_mat@y[self.ndofs:]
                                                     + Ft)))

    def evaluateAt(self, y, position):
        from dune.common import FieldVector
        from dune.fem.utility import pointSample
        from dune.generator import algorithm
        # convert vector back to DUNE function that can be sampled on the mesh
        y1 = np.array(y[:self.ndofs])
        df_y = self.V.function("df_y1", dofVector=y1 )

        if len(position) < self.mesh.dimension:
            position.append(0)

        val = pointSample( df_y, position )
        return val[ 1 ] # return displacement in y-direction

    def plotBeam( self, y ):
        # convert vector back to DUNE function
        y1 = np.array(y[:self.ndofs])
        displacement = self.V.function("displacement", dofVector=y1 )
        from dune.fem.view import geometryGridView
        x = SpatialCoordinate(self.V)
        # interpolate into coordinates for geometryGridView
        position = self.V.interpolate( x+displacement, name="position" )
        beam = geometryGridView( position )
        beam.plot()

if __name__ == '__main__':
    # test section using build-in ODE solver from Assimulo
    t_end = 8
    beam_class = elastodynamic_beam(4, T=t_end)

    import assimulo.solvers as aso
    import assimulo.ode as aode

    # Use a separate zero damping matrix for CVODE comparison
    C_0 = ssp.csc_matrix(beam_class.Dampening_mat.shape)

    def rhs_cvode(t, y):
        Ft = t*beam_class.F if t < beam_class.cutoff_Tc else np.zeros(beam_class.ndofs)
        return np.hstack((y[beam_class.ndofs:],
                          ssl.spsolve(beam_class.Mass_mat,
                                      -beam_class.Stiffness_mat@y[:beam_class.ndofs]
                                      -C_0@y[beam_class.ndofs:]
                                      + Ft)))

    # y , ydot
    beam_problem = aode.Explicit_Problem(rhs_cvode, y0=np.zeros((2*beam_class.ndofs,)))
    beam_problem2 = EP2.Explicit_Problem_2nd(beam_problem, n=beam_class.ndofs)
    beam_problem2.name='Modified Elastodyn example from DUNE-FEM'
    
    beamCV = aso.ImplicitEuler(beam_problem) # CVode solver instance
    #beamCV = aso.Radau5ODE(beam_problem)
    beamCV.h = 0.05 # constant step size here
    tt, y = beamCV.simulate(t_end)

    disp_tip = []
    plottime = 0
    plotstep = 0.25
    for i, t in enumerate(tt):
        disp_tip.append(beam_class.evaluateAt(y[i], [1, 0.05]))
        if t > plottime:
            print(f"Beam position at t={t}")
            #beam_class.plotBeam( y[i] )
            plottime += plotstep

    disp_tip_cvode = np.array(disp_tip)

    pl.figure()
    pl.plot(tt, disp_tip, '-b')
    pl.title('Displacement of beam tip over time (CVODE)')
    pl.xlabel('t')
    pl.savefig('displacement_CVODE.png', dpi = 200)

    # Define K, C M and f
    K, C, M = beam_class.Stiffness_mat, beam_class.Dampening_mat, beam_class.Mass_mat, 

    def force(t):
        # return beam_class.F
        # Apply time-dependent force: ramps up linearly during cutoff_Tc, then constant
        Ft = t*beam_class.F if t < beam_class.cutoff_Tc else np.zeros(beam_class.ndofs)
        return Ft
    
    # Create Instance of NewmarkBetaSolver
    beamCV = BetaSolver.NewmarkBetaSolver(beam_problem2, beta, gamma, K, C, M, force)
    #beamCV = aso.Radau5ODE(beam_problem)
    beamCV.h = 0.05 # constant step size here
    ttN, yN = beamCV.simulate(t_end)
    
    # Plot
    disp_tip = []
    plottime = 0
    plotstep = 0.25
    for i, t in enumerate(ttN):
        disp_tip.append(beam_class.evaluateAt(yN[i], [1, 0.05]))
        if t > plottime:
            print(f"Beam position at t={t}")
            #beam_class.plotBeam( yN[i] )
            plottime += plotstep

    disp_tip_newmark = np.array(disp_tip)

    pl.figure()
    pl.plot(ttN, disp_tip, '-b')
    pl.title('Displacement of beam tip over time (Newmark-β)')
    pl.xlabel('t')
    pl.savefig('displacement_Newmark.png', dpi = 200)
    
    # Create Instance of HHTalphaSolver
    beamCV = HHTSolver.HHTalphaSolverElasto(beam_problem2, alpha, K, C, M, force)
    #beamCV = aso.Radau5ODE(beam_problem)
    beamCV.h = 0.05 # constant step size here
    ttH, yH = beamCV.simulate(t_end)

    # Plot
    disp_tip = []
    plottime = 0
    plotstep = 0.25
    for i, t in enumerate(ttH):
        disp_tip.append(beam_class.evaluateAt(yH[i], [1, 0.05]))
        if t > plottime:
            print(f"Beam position at t={t}")
            #beam_class.plotBeam( yH[i] )
            plottime += plotstep

    disp_tip_hht = np.array(disp_tip)

    pl.figure()
    pl.plot(ttH, disp_tip, '-b')
    pl.title('Displacement of beam tip over time (HHT-α)')
    pl.xlabel('t')
    pl.savefig('displacement_HHT.png', dpi = 200)

    pl.figure()
    pl.plot(tt, disp_tip_cvode, label='CVODE ImplicitEuler (C_0)', linewidth=2)
    pl.plot(ttN, disp_tip_newmark, label='Newmark-β (C)', linewidth=2)
    pl.plot(ttH, disp_tip_hht, label='HHT-α (C)', linewidth=2)
    pl.title('Beam Tip Displacement Comparison')
    pl.xlabel('t')
    pl.ylabel('tip displacement (y)')
    pl.grid(True)
    pl.legend()
    pl.tight_layout()
    pl.savefig('displacement_combined.png', dpi=200)
    
    print(f"Max absolute error: {np.max(np.abs(yN - yH)):.2e}")
    print(f"Mean absolute error: {np.mean(np.abs(yN - yH)):.2e}")
    print(f"L2 error norm: {np.linalg.norm(yN - yH):.2e}")