import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np 
import matplotlib.pyplot as plt
from src.engines.black_scholes import time_bs_wrapper

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

if __name__ == "__main__":
    N_values = [10, 20, 40, 80, 160]
    plot_convergence(time_bs_wrapper, N_values)