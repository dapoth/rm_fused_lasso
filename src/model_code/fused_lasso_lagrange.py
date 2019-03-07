import cvxpy as cp
import numpy as np

def fused_lasso_lagrange(y, X, lambda1, lambda2):
    """Solves for given data and penalty constants the fused lasso lagrange form.

    Args:
        | y (np.ndarray): 1d array of dependent variables
        | X (np.ndarray): 2d array of independent variables
        | s1 (float): constraint on ||b||_1
        | s2 (float): constraint on the absolute jumps in beta

    Returns:
        beta.value (np.ndarray)

    """
    if len(y) != len(X):
        raise TypeError("The length of y must be equal to the number of rows of x.")

    if np.size(lambda1) > 1 or np.size(lambda2) > 1:
        raise TypeError("The penalty constants need to have length one.")

    if lambda1 < 0 or lambda2 < 0:
        raise ValueError("The penalty constants need to be nonnegative.")

    n_features = len(X[1, :])
    beta_hat = cp.Variable(n_features)
    error = cp.sum_squares(X*beta_hat - y)
    obj = cp.Minimize(error + lambda1*cp.norm(beta_hat, 1) + lambda2*
                      cp.norm(beta_hat[1:n_features]-beta_hat[0:n_features-1], 1))
    prob = cp.Problem(obj)
    prob.solve()

    return beta_hat.value
