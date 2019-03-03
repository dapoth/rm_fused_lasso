import cvxpy as cp

def fused_lasso_dual(y, x, lambda1, lambda2):
    """
    Solves for given data and penalty constants the fused lasso dual form.
    """
    
    if len(y) != len(x):
        raise ValueError("The length of y must be equal to the number of rows of x.") 
    #if lambda1 < 0 | lambda2 < 0:
    #    raise ValueError("The penalty constraints need to be nonnnegative.")
        
    p = len(x[1, :])
    b = cp.Variable(p)
    error = cp.sum_squares(x*b - y)
    obj = cp.Minimize(error + lambda1 * cp.norm(b, 1) +
                      lambda2 * cp.norm(b[1:p]-b[0:p-1], 1))
    prob = cp.Problem(obj)
    prob.solve()

    return b.value
