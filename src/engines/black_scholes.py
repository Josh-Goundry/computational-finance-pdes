import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib.pyplot as plt
from src.solvers.tridiagonal import thomas_solver as ts

def bs_solver(N, M, f_0, f_R, g, vol, r, T, K):
    '''
    Solver for the backward time Black-Scholes equation.

    ========INPUTS========
    N : int
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
    x : np.ndarray, shape(N+1,)
        Full spacial grid from x_0=0.0 to x_N=1.0 with step size h.
    t : np.ndarray, shape(M+1,)
        Full temporal grid from t_0=0.0 to t_M=T with step size dt.
    '''

    