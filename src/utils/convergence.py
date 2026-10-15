import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np 
import matplotlib.pyplot as plt
from src.engines.heat_diffusion import heat_eq

def plot_convergence(fn_wrapper, N_values):
    '''
    Function that plots the L2 and L_inf errors of the called function wrapper for the entries in N_values.
    
    ========INPUTS========
    fn_wrapper : function
        The wrapper providing our numerical and exact solutions with their error. Must return step size, L2 error, and L_inf error.
    N_values: np.ndarray or list
        The number of steps (1/step size) we iterate over.

    ========OUTPUTS========
    p : float
        Our log slope coefficient.
    dx_array : np.ndarray
        The array of step sizes plotted.
    error_array : np.ndarray
        The array of all L_inf errors used to calculate p.
    '''

    #Defining our lists to be filled with error values
    step_size, L2_error, L_inf_error = [], [], []

    #Filling our lists with the error at each N value
    for N in N_values:
        dx, L2, L_inf = fn_wrapper(N)
        step_size.append(dx)
        L2_error.append(L2)
        L_inf_error.append(L_inf)

    #Plotting our errors on a log-log graph against the step size
    plt.loglog(step_size, L2_error, marker='o', color='red', label='$L_2$ Error')
    plt.loglog(step_size, L_inf_error, marker='*', color='blue', label='$L_∞$ Error')

    #Finding the slope p
    dx_array = np.array(step_size)
    error_array = np.array(L_inf_error) #Can be changed for L_inf_error for inf norm calculation

    p, intercept = np.polyfit(np.log(dx_array), np.log(error_array), 1)

    print(f"The slope of our log relation is {p:.2f}, proving {p} convergence.")

    plt.xlabel('Step size')
    plt.ylabel("Error")
    plt.legend()
    plt.show()

    return p, dx_array, error_array

# Copy of the wrapper in `02_Heat_Equation_Verification.ipynb` 
def time_heat_wrapper(M):
    #Assigning parameters
    a = np.pi**-2
    T = 1
    N = 10000
    alpha = 0
    beta = 1
    u_0 = lambda x: np.sin(2*np.pi*x) + x
    f = None

    #Defining our step sizes
    dx = 1/N
    dt = T/M
    
    #Asigning our exact and numerical solution
    u_num, x, t = heat_eq(f, N, M, alpha, beta, u_0, a, T)
    u_exact = np.exp(-4*T) * np.sin(2*np.pi*x) + x

    #Finding Error
    L_inf_error = np.max(np.abs(u_num[:, -1] - u_exact))
    L2_error = np.sqrt(dx*np.sum((u_num[:, -1] - u_exact)**2))

    return dt, L2_error, L_inf_error

M_values = [10, 20, 40, 80, 160]

if __name__ == "__name__":
    plot_convergence(time_heat_wrapper, M_values)