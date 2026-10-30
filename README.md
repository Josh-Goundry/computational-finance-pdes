# computational-finance-pdes
A Python suite of Finite Difference Method solvers for 1D PDEs. From steady state, elliptic BVPs and the Heat Equation to an algorithmic Black-Scholes Option Pricing Engine (European options).

## Overview
This repository provides numerical implementations for physics and quantitative finance differential equations. It implements:
 - **Elliptic BVPs**: Solved using a standard central difference scheme (CDM).
 - **Parabolic PDEs (Black-Scholes and Heat equation)**: Solved using implicit time-stepping methods to ensure L-stability.
 - **Linear Algebra**: Custom script to solve the tridiagonal system Au=b using the Thomas Algorithm.

## Repo Structure
 - `notebooks/`: Interactive Jupyter notebooks for verification, derivation, and error plots.
    - `01_Elliptic_Verification.ipynb`: Examining the CDM elliptic BVP.
    - `02_Heat_Equation_Verification.ipynb`: Examining the BTCS scheme for the Heat equation.
    - `03_Black_Scholes_Verification.ipynb`: Examining the BTCS scheme for the forward time-to-maturity Black-Scholes model.
    - `04_Rannacher_Smoothing_Crank_Nicolson.ipynb`: Comparing Rannacher Smoothed and Naive Crank-Nicolson implementation for the forward time-to-maturity Black-Scholes model. 
 - `src/`: Main suite of solvers
    - `solvers/`: Low level matrix solvers (`tridiagonal.py`, `elliptic.py`)
    - `engines/`: Time dependent PDE solvers (`heat_diffusion.py`, `black_scholes.py`)
    - `utils/` : Verification Tools (`greeks.py`, `convergence.py`)

## Numerical Methods
### Steady-State Elliptic Solvers
Discretises the spatial derivative using the second order central difference approximation:
$$u''(x_i) \approx \frac{u_{i-1} - 2u_i + u_{i+1}}{\Delta x^2}$$

### Implicit Time Dependent PDEs
To maintain stability and avoid stiffness, the parabolic PDEs have been discretised using the implicit Backward Euler scheme for L-Stability, transforming each step into a tridiagonal system of the form $A u^{n+1} = b$.

### Further Schemes
To examine higher order convergence in time, both Rannacher Smoothed and Naive Crank-Nicolson are implemented for the Black-Scholes model using flags to select the scheme of choice ("BTCS", "CN", "RS"), exploring the benefits and limitations of each. 

## Getting Started

### Requirements
 - Python 3.9+
 - Dependencies: `numpy`, `scipy`, `matplotlib`, `jupyter`

### Installation

**Clone the Repository**
```bash
git clone [https://github.com/Josh-Goundry/computational-finance-pdes.git](https://github.com/Josh-Goundry/computational-finance-pdes.git)
cd computational-finance-pdes
```

**Install Dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### Running the Engines via Python
You can import the pricing engine directly into your scripts:

```python
from src.engines.black_scholes import bs_solver

#Initialise the solver
V, s, t = bs_solver(R, M, f_0, f_R, g, vol, r, T, K, method)
```

### Launching the Interactive Notebooks
To launch the notebooks showing convergence plots, error analysis, and derivations:
```bash
jupyter notebook
```
Navigate to `notebooks/` for the notebooks listed under Repo Structure.

## Verification and Accuracy
The utilities in `src/utils/` show and confirm the spatial and temporal accuracy of the finite difference methods used:
 - **Order of Convergence (`convergence.py`)**: Validates $\mathcal{O}(\Delta x^2)$ spatial accuracy for the CDM scheme, and $\mathcal{O}(\Delta t)$ temporal accuracy for Backward Euler.
 - **Greeks Calculation (`greeks.py`)**: Computes Delta, Gamma, and Theta numerically using the central differences on the solved option price surface. 

## References
Giles, M. B., & Carter, R. (2006). *Convergence analysis of Crank–Nicolson and Rannacher time-marching. *The Journal of Computational Finance*, 9(4), 89–112. [Available online](https://people.maths.ox.ac.uk/gilesm/files/giles_carter.pdf)

## Acknowledgement
I would also like to extend a big thank you to Dr. Kris van der Zee of the School of Mathematics at the University of Nottingham for recommending this project and sharing MATH4064 course details, allowing me to explore foundational PDE discretisation and numerical convergence analysis concepts.