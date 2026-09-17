import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib.pyplot as plt
from src.solvers.tridiagonal import thomas_solver as ts
from scipy import special

def bs_solver(R, M, f_0, f_R, g, vol, r, T, K):
    '''
    Solver for the backward time Black-Scholes equation.

    ========INPUTS========
    R : int
        The number of spacial grid points (call option prices). Creates N+1 grid points and N-1 unknowns.
    M : int
        The number of temporal grid points. Creates M+1 grid points from t_0 to t_M.
    f_0 : callable
        Boundary value function at V(t,0). Must accept an array of t coordinates.
    f_R : callable
        Boundary value function at V(t,R). Must accept an array of t coordinates.
    g : callable
        Initial condition at V(0,S). Must accept an array of x coordinates.
    vol : float
        The volatility coefficient of the diffusion term (Gamma).
    T : float, default=1.0
        Time until expiration.
    K : float
        Strike price of the call option.

    ========OUTPUT========
    V : ndarray(), shape(N+1, M+1)
        Complete numerical approximation including boundary values and IC.
    s : np.ndarray, shape(N+1,)
        Full spacial grid from x_0=0.0 to x_N=1.0 with step size h.
    t : np.ndarray, shape(M+1,)
        Full temporal grid from t_0=0.0 to t_M=T with step size dt.
    '''

    #Defining the grid spaces and grids
    h = 3*K/R
    dt = T/M

    s = np.linspace(0.0, 3*K, R+1)
    t = np.linspace(0.0, T, M+1)

    #Creating the interal grid and the matrix V
    s_int = s[1:-1]
    V = np.zeros((R+1, M+1))
    i = np.arange(1, R)

    #Defining the tridiagonal entries
    sub_diag = -(dt/2)*(vol**2)*i**2
    sub_diag[0] = 0
    main_diag = (1 + dt*(vol**2 * i**2 + r*i + r))
    super_diag = -dt*((vol**2 * i**2)/2 + r*i)
    super_diag[-1] = 0

    #Appling our BCs and IC
    V[0,:] = f_0(t)         #BC_0
    V[-1,:] = f_R(t, s[-1])     #BC_1
    V[:,0] = g(s)           #IC

    #Recursive loop solving each tridiagonal system
    for n in range(M):
        b = V[1:-1, n].copy()
        b[0] += (dt/2*(vol**2)*1) * f_0(t[n+1])
        b[-1] += dt*((vol**2 * (R-1)**2)/2 + r*(R-1)) * f_R(t[n+1], s[-1])

        V[1:-1,n+1] = ts(sub_diag, main_diag, super_diag, b)
    
    return V, s, t

if __name__ == "__main__":
    K = 100
    M = 20
    r = 0
    vol = 0.5
    T = 5
    R = 300
    f_0 = lambda t: K*np.exp(-r*t)
    d_plus = lambda t, s: 1/(vol*np.sqrt(t)) * (np.log(s/K) + (r + ((vol**2)/2)*t))
    d_minus = lambda t, s: 1/(vol*np.sqrt(t)) * (np.log(s/K) + (r - ((vol**2)/2)*t))
    phi = lambda s: 1/2 * special.erfc(-s/np.sqrt(2))
    f_R = lambda t, s: K*np.exp(-r*t)*phi(-d_minus(t, s))-s*phi(-d_plus(t, s))
    g = lambda s: np.maximum(K-s, 0)
    V, s, t = bs_solver(R, M, f_0, f_R, g, vol, r, T, K)

    S_grid, T_grid = np.meshgrid(s, t)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(S_grid, T_grid, V.T, cmap='viridis', edgecolor=None)

    ax.set_xlabel('Spot Price (S)')
    ax.set_ylabel('Time (t)')
    ax.set_zlabel('Option Price (V)')
    ax.set_title('Black-Scholes Option Surface')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='V(S, t)')

    plt.tight_layout()
    plt.show()