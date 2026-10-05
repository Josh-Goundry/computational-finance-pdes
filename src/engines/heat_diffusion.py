import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib.pyplot as plt
from src.solvers.tridiagonal import thomas_solver as ts

def heat_eq(f, N, M, alpha, beta, u_0, a, T):
    '''
    Discretises and solves the system du/dt = a*d^2u/dx^2 + f(x,t)

    ========INPUTS========
    f : callable or None
        The source function f(x, t). Must accept spacial coordinates x and temporal coordinates t.
    N : int
        The number of spacial grid points. Creates N+1 grid points and N-1 unknowns. 
    M : int
        The number of time steps, creating M+1 grid points from t_0 to t_M. 
    alpha: float
        Dirichlet boundary condition at u(0, t)=alpha.
    beta: float
        Dirichlet boundary condition at u(1, t)=beta.
    u_0: callable
        Initial condition at u(x, 0). Must accept 
    a : float
        The diffusion coefficient. 
    T : float, default = 1.0
        The total simulation time.


    ========OUTPUTS========
    u : np.ndarray, shape(N+1, M+1)
        Complete numerical solution including boundary vaues and ICs.
    x : np.ndarray, shape(N+1,)
        Full spacial grid from x_0=0.0 to x_N=1.0 with step size h.
    t : np.ndarray, shape(M+1,)
        Full temporal grid from t_0=0.0 to t_M=T with step size dt.
    '''

    #Setting up the step sizes and grids
    h = 1.0/N
    dt = T/M

    x = np.linspace(0.0, 1.0, N+1)
    t = np.linspace(0.0, T, M+1)

    #Defining internal grid
    x_int = x[1:-1]

    #Defining lambda and u
    lam_val = (a*dt)/h**2
    u = np.zeros((N+1, M+1))

    #Implementing the IC and BCs
    u[:,0] = u_0(x)
    u[0,:] = alpha
    u[-1,:] = beta

    #Defining our tridiagonal arrays
    sub_diag = np.full(N-1, -lam_val, dtype=float)
    sub_diag[0] = 0
    main_diag = np.full(N-1, 1+2*lam_val, dtype=float)
    super_diag = np.full(N-1, -lam_val, dtype=float)
    super_diag[-1] = 0

    #Recursive tridiagonal solver and defining the RHS
    for n in range(M):
        b = u[1:-1, n].copy()
        b[0] += lam_val*alpha
        b[-1] += lam_val*beta

        #Summing our forcing function f
        if f is not None:
            b += dt*f(x_int, n+1)

        #Solving the system Au = b
        u[1:-1, n+1] = ts(sub_diag, main_diag, super_diag, b)
    
    return u, x, t

def space_heat_wrapper(N):
    #Defining our step size
    dx = 1.0/N

    #Assigning parameters
    a = np.pi**-2
    T = 1
    alpha = 0
    beta = 1
    u_0 = lambda x: np.sin(2*np.pi*x) + x
    M = 1000000 #Maintianing a large temporal step count for minimal error interference
    f = None

    #Asigning our exact and numerical solution
    u_num, x, t = heat_eq(f, N, M, alpha, beta, u_0, a, T)
    u_exact = np.exp(-4*T) * np.sin(2*np.pi*x) + x

    #Finding Error
    L_inf_error = np.max(np.abs(u_num[:, -1] - u_exact))
    L2_error = np.sqrt(dx*np.sum((u_num[:, -1] - u_exact)**2))
    
    return dx, L2_error, L_inf_error

def time_heat_wrapper(M):
    #Assigning parameters
    a = np.pi**-2
    T = 1
    alpha = 0
    beta = 1
    u_0 = lambda x: np.sin(2*np.pi*x) + x
    N = 10000 #Maintianing a large spatial step count for minimal error interference
    f = None

    #Defining our step size
    dt = T/M
    
    #Asigning our exact and numerical solution
    u_num, x, t = heat_eq(f, N, M, alpha, beta, u_0, a, T)
    u_exact = np.exp(-4*T) * np.sin(2*np.pi*x) + x

    #Finding Error
    L_inf_error = np.max(np.abs(u_num[:, -1] - u_exact))
    L2_error = np.sqrt(dt*np.sum((u_num[:, -1] - u_exact)**2))

    return dt, L2_error, L_inf_error

if __name__ == "__main__":
    a = np.pi**-2
    T = 1
    alpha = 0
    beta = 1
    u_0 = lambda x: np.sin(2*np.pi*x) + x
    N = 100
    M = 100
    f = None
    u, x, t = heat_eq(f, N, M, alpha, beta, u_0, a, T)

    X, T_grid = np.meshgrid(x, t)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, T_grid, u.T, cmap='viridis', edgecolor=None)

    ax.set_xlabel('Space (x)')
    ax.set_ylabel('Time (t)')
    ax.set_zlabel('Temperature (u)')
    ax.set_title('3D Heat Diffusion Solution $u(x, t)$')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='u(x, t)')

    plt.tight_layout()
    plt.show()