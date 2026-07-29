import numpy as np

def thomas_solver(a, b, c, d):
    '''
    Thomas Algorithm Solver for tridiagonal systems of the form Ax=d where A is a Matrix of dimensions nxn and d is a vector of dimensions nx1.
    
    a: Array correlating to the superdiagonal entries of length n-1
    b: Array correlating to the diagonal entries of length n
    c: Array Correlating to the subdiagonal entries of length n-1
    d: Our Target vector
    x: Our solution

    The aim is to use the Thomas algorithm to solve the system in O(n) time rather than O(h^3) using standard LU factorisation.
    '''

    #Length of our matrix/vector
    n=len(d)
    
    #Our solution array
    x=np.zeroes(n)
    
    #Sub, super, and main diagonal + vector arrays
    a=np.array(a, dtype=float)
    b=np.array(b, dtype=float)
    c=np.array(c, dtype=float)
    d=np.array(d, dtype=float)

    '''
    TODO: 
    LU factorisation
    Forward substition
    Backward substitution
    '''

    return x