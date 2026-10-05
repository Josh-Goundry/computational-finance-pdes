import numpy as np
import matplotlib.pyplot as plt
from src.solvers.tridiagonal import thomas_solver

def elliptic_bvp_solver(alpha, beta, f, N):
    '''
    Discretizes and solves the 1D Epilleptic BVP -u''(x)=f(x) on (0,1)

    ========INPUTS========
    alpha : float
        Dirichlet boundary condition u(0)=alpha
    beta : float
        Dirichlet boundary condition u(1)=beta
   f : callable
        The right hand side source function f(x). Must accept a NumPy array of x-coordinates 
    N : int
        Number of spacial intervals. Creates N+1 grid points and N-1 unknowns

    ========OUTPUTS========
    x : np.ndarray, shape(N+1,)
        Full spacial coordinates from x_0=0.0 to x_N=1.0
    u : nd.array, shape(N+1, )
        Complete numerical solution including boundary values
    '''
    #Grid setup
    h = 1.0/N
    x = np.linspace(0.0, 1.0, N+1)

    #Assign solution array and enforce DBCs
    u = np.zeros(N+1, dtype=float)
    u[0] = alpha
    u[N] = beta

    #No. internal unknowns
    n_int = N-1
    
    #Internal points array
    x_int = x[1:-1]

    #Defining tridiagonal arrays
    b_diag = np.full(n_int, 2.0, dtype=float)
    #Include padding for non main diagonals
    a_diag = np.zeros(n_int, dtype=float)
    a_diag[1:] = -1.0

    c_diag = np.zeros(n_int, dtype=float)
    c_diag[:-1] = -1.0

    #Assmebling RHS vector  
    d = (h**2)*f(x_int)
    d[0] += alpha
    d[-1] += beta

    #Solving the tridiagonal system and reassemble u(x)
    u_int = thomas_solver(a_diag, b_diag, c_diag, d)
    u[1:-1] = u_int
    
    return x, u

if __name__ == "__main__":
    alpha = 0
    beta = 1
    u_exact = lambda x: x - np.sin(2*np.pi*x)
    f = lambda x: -4*(np.pi**2)*np.sin(2*np.pi*x)
    N = 10
    x, u = elliptic_bvp_solver(alpha, beta, f, N)
        
    #Plot against true solution
    plt.figure(figsize=(8,5))
    
    plt.plot(x, u, '-', label=f'Numerical Approximation N={N}')
    
    x_fine = np.linspace(0.0, 1.0, 200)
    plt.plot(x_fine, u_exact(x_fine), label='True Solution: $u_{exact}(x)$')

    plt.title("1D Elliptic BVP: $-u''(x) = f(x)$")
    plt.xlabel("$x$")
    plt.ylabel("$u(x)$")
    plt.legend()   
    plt.grid(True)
    plt.show()
    