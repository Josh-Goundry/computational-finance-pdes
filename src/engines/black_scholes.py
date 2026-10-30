import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib.pyplot as plt
from src.solvers.tridiagonal import thomas_solver as ts
from scipy import special

def bs_solver(R, M, f_0, f_R, g, vol, r, T, K, method='scheme'):
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
    method : "BTCS", "CN", "RS"
        Choose the scheme which solves the problem. RS = Rannacher Smoothed Crank-Nicolson, CN = Naive Crank-Nicolson, BTCS = 

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
    V = np.zeros((R+1, M+1))
    i = np.arange(1, R)

    #Defining the tridiagonal entries for BTCS
    sub_diag = (dt/2) * i * (r - (vol**2 * i))
    sub_diag[0] = 0
    main_diag = (1 + dt*(vol**2 * i**2 + r))
    super_diag = -(dt/2) * i * (r + (vol**2 * i))
    super_diag[-1] = 0

    #Defining the tridiagonal entries for CN
    a_cn = (dt * vol**2 * i**2)/4
    b_cn = (dt * r * i)/4
    c_cn = (dt * r)/2

    cn_sub = (b_cn - a_cn).copy()
    cn_sub[0] = 0
    cn_main = 1 + 2*a_cn + c_cn
    cn_super = -(a_cn + b_cn).copy()
    cn_super[-1] = 0

    #Appling our BCs and IC
    V[0,:] = f_0(t)             #BC_0
    V[-1,:] = f_R(t, s[-1])     #BC_1
    V[:,0] = g(s)               #IC

    #Recursive loops solving each tridiagonal system
    if method == "BTCS":
        for n in range(M):
            b = V[1:-1, n].copy()
            b[0] -= (dt/2) * 1 * (r - (vol**2 * 1)) * f_0(t[n+1])
            b[-1] -= -(dt/2) * (R-1) * (r + (vol**2 * (R-1))) * f_R(t[n+1], s[-1])

            V[1:-1,n+1] = ts(sub_diag, main_diag, super_diag, b)

    if method == "CN":
        for n in range(M):
            b_temp = V[1:-1, n].copy()
            b = np.zeros(R-1)
            
            b[:-1] += b_temp[1:]*(a_cn[:-1] + b_cn[:-1])
            b[1:] += b_temp[:-1]*(a_cn[1:] - b_cn[1:])
            b += b_temp*(1-2*a_cn-c_cn)

            b[0] += (a_cn[0] - b_cn[0])*(f_0(t[n]) + f_0(t[n+1]))
            b[-1] += (a_cn[-1] + b_cn[-1])*(f_R(t[n], s[-1]) + f_R(t[n+1], s[-1]))

            V[1:-1,n+1] = ts(cn_sub, cn_main, cn_super, b)

    if method == "RS":
        half_dt = dt / 2
        b_half = V[1:-1, 0]

        sub_diag_half = (half_dt/2) * i * (r - (vol**2 * i))
        sub_diag_half[0] = 0
        main_diag_half = (1 + half_dt*(vol**2 * i**2 + r))
        super_diag_half = -(half_dt/2) * i * (r + (vol**2 * i))
        super_diag_half[-1] = 0

        #Backward Euler
        for n in range(1, 5):
            t_half = n * half_dt

            b = b_half.copy()
            b[0] -= (half_dt/2) * 1 * (r - (vol**2 * 1)) * f_0(t_half)
            b[-1] -= -(half_dt/2) * (R-1) * (r + (vol**2 * (R-1))) * f_R(t_half, s[-1])

            b_half = ts(sub_diag_half, main_diag_half, super_diag_half, b)

            if n % 2 == 0:
                V[1:-1, n//2] = b_half

        #Crank-Nicolson
        for n in range(2, M):
            b_temp = V[1:-1, n].copy()
            b = np.zeros(R-1)
            
            b[:-1] += b_temp[1:]*(a_cn[:-1] + b_cn[:-1])
            b[1:] += b_temp[:-1]*(a_cn[1:] - b_cn[1:])
            b += b_temp*(1-2*a_cn-c_cn)

            b[0] += (a_cn[0] - b_cn[0])*(f_0(t[n]) + f_0(t[n+1]))
            b[-1] += (a_cn[-1] + b_cn[-1])*(f_R(t[n], s[-1]) + f_R(t[n+1], s[-1]))

            V[1:-1,n+1] = ts(cn_sub, cn_main, cn_super, b)
    return V, s, t

if __name__ == "__main__":
    method = "RS"
    K = 100
    M = 20
    r = 0
    vol = 0.5
    T = 5
    R = 300
    f_0 = lambda t: K*np.exp(-r*t)
    d_plus = lambda t, s: 1/(vol*np.sqrt(np.maximum(t, 1e-14))) * (np.log(np.maximum(s, 1e-15)/K) + (r + ((vol**2)/2)*np.maximum(t, 1e-14)))
    d_minus = lambda t, s: 1/(vol*np.sqrt(np.maximum(t, 1e-14))) * (np.log(s/K) + (r - ((vol**2)/2)*np.maximum(t, 1e-14)))
    phi = lambda s: 1/2 * special.erfc(-s/np.sqrt(2))
    f_R = lambda t, s: K*np.exp(-r*t)*phi(-d_minus(t, s))-s*phi(-d_plus(t, s))
    g = lambda s: np.maximum(K-s, 0)
    V, s, t = bs_solver(R, M, f_0, f_R, g, vol, r, T, K, method)

    S_grid, T_grid = np.meshgrid(s, t)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(S_grid, T_grid, V.T, cmap='viridis', edgecolor=None)

    ax.set_xlabel('Spot Price (S)')
    ax.set_ylabel('Time (t)')
    ax.set_zlabel('Option Price (V)')
    ax.set_title('Black-Scholes Option Surface (' + str(method) + ')')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='V(S, t)')

    plt.tight_layout()
    plt.show()