import numpy

'''
A program that calculates the greeks Gamma, Delta, and Theta using the output from the black scholes solver.

========INPUTS========
V : np.ndarray, shape(R+1, M+1)
    The output matrix from the backward time black scholes solver.
R : int
    The number of spatial steps within the grid from 0.0 to 3K used to calculate the step size h.
M : int
    The number of temporal steps within the grid from 0.0 to T used to calculate the step size dt.
T : float  
    The time until expiration in the solver.
K : float
    The option strike price.

========OUTPUTS========
delta : np.ndarray, shape(R-1, )
    The estimate of delta using from V.
gamma : np.ndarray, shape(R-1, )
    The estimate of gamma from V.
theta : np.ndarray, shape(R-1, )
    The estimate of theta from V.
'''

def delta(V, R, K):
    h = 3*K/R

    #Defined using the first order central difference approximation
    delta = (V[2:, -1] - V[:-2, -1])/(2*h)

    return delta

def gamma(V, R, K):
    h = 3*K/R

    #Defined using the second order central difference approximation
    gamma = (V[2:, -1] - 2*V[1:-1, -1] + V[:-2, -1])/h**2
    
    return gamma

def theta(V, T, M):
    dt = T/M

    #Defined using the backwards difference approximation
    theta = (V[1:-1, -2] - V[1:-1, -1])/dt
    
    return theta