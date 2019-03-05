import cvxpy as cp
import numpy as np

def fused_lasso_primal(y, X, s1, s2):
    """Compute fused lasso estimates for given penalty constraints s1 and s2.

    Args:
        y (np.ndarray): 1d array of dependent variables
        X (np.ndarray): 2d array of independent variables
        s1 (float): constraint on ||b||_1
        s2 (float): constraint on the absolute jumps in beta

    Returns:
        beta.value (np.ndarray)

    """
    if len(y) != len(X):
        raise TypeError("The length of y must be equal to the number of rows of x.") 
        
    if np.size(s1) > 1 or np.size(s2) > 1:
        raise TypeError("The penalty constants need to have length one.")
        
    if s1 < 0 or s2 < 0:
        raise ValueError("The penalty constants need to be nonnnegative.")

    n_features = len(X[1, :])
    beta = cp.Variable(n_features)
    error = cp.sum_squares(X * beta - y)
    obj = cp.Minimize(error)
    constraints = ([cp.norm(beta, 1) <= s1,
                    cp.norm(beta[1:n_features]-beta[0:n_features-1], 1) <= s2])
    prob = cp.Problem(obj, constraints)
    prob.solve()

    return beta.value
