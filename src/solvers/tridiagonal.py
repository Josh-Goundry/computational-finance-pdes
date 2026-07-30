import numpy as np

def thomas_solver(a, b, c, d):
    '''
    Thomas Algorithm Solver for tridiagonal systems of the form Ax=d where A is a Matrix of dimensions nxn and d is a vector of dimensions nx1.
    
    ========INPUTS========
    a : array_like, shape(n,),
        Subdiagonal entries (lower band). Note that a[0] is padded/unused.
    b : array_like, shape(n,),
        Main Diagonal entries.
    c : array_like, shape(n,),
        Superdiagonal entries (upper band). Note that c[-1] is padded/unused.
    d : array_like, shape(n,),
        Target vector entries.

    ========OUTPUTS========
    x : np.ndarray, shape(n,),
        Solution to the system Ax=d

    ========NOTE========
    Solves the system using the Thomas algorithm in O(n) time and memory, bypassing the O(h^3) complexity of standard LU factorisation.
    '''

    #Length of our matrix/vector
    n=len(d)
    
    #Our solution array
    x=np.zeros(n)
    
    #Sub, super, and main diagonal + vector arrays
    a=np.array(a, dtype=float)
    b=np.array(b, dtype=float)
    c=np.array(c, dtype=float)
    d=np.array(d, dtype=float)

    #LU Factorisation and Forward Substitution
    for i in range(1,n):
        m=a[i]/b[i-1]
        b[i]=b[i]-m*c[i-1]
        d[i]=d[i]-m*d[i-1]
    
    #Backward Substitution to solve the final system Ux=y for x
    x[n-1]=d[n-1]/b[n-1]
    for i in range(n-2,-1,-1):
        x[i]=(d[i]-c[i]*x[i+1])/b[i]

    return x

if __name__ == "__main__":
    test_a = [0.0, -2.0, -4.0]
    test_b = [2.0,  3.0,  5.0]
    test_c = [-1.0, -1.0,  0.0]
    test_d = [0.0,  1.0,  7.0]

    print("Solution:", thomas_solver(test_a, test_b, test_c, test_d))
