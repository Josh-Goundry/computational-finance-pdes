import numpy as np
import matplotlib.pyplot as plt

def elliptic_bvp_solver(alpha, beta, f, N):
    '''
    Discretizes and solves the 1D Epilleptic BVP -u''(x)=f(x) on (0,1)

    ========INPUTS========
    f : callable
        The right hand side source function f(x). Must accept a NumPy array of x-coordinates
    alpha : float
        Dirichlet boundary condition u(0)=alpha
    beta : float
        Dirichlet boundary condition u(1)=beta
    N : int
        Number of spacial intervals. Creates N+1 grid points and N-1 unknowns

    ========OUTPUTS========
    x : np.ndarray, shape(N+1,)
        Full spacial coordinates from x_0=0.0 to x_N=1.0
    u : nd.array, shape(N+1, )
        Complete numerical solution including boundary values
    '''
    #Grid setup
    h=1.0/N
    x=np.linspace(0.0,1.0,N+1)

    #Assign solution array and enforce DBCs
    u=np.zeros(N+1, dtype=float)
    u[0]=alpha
    u[N]=beta

    #No. internal unknowns
    n_int=N-1

    '''
    TODO:
    1. Separate interior points
    2. Assemble Tridiagonal Arrays
    3. Assemble RHS vector and apply DBCs
    4. Solve and reassemble u(x)
    5. Generate plot of u on the linespace x
    '''
    
    return x,u