import cvxpy as cp

def fused_lasso_dual(y, X, lambda1, lambda2):
    """Solves for given data and penalty constants the fused lasso dual form.

    Args:
        y (np.ndarray): 1d array of dependent variables
        X (np.ndarray): 2d array of independent variables
        s1 (float): constraint on ||b||_1
        s2 (float): constraint on the absolute jumps in beta

    Returns:
        beta.value (np.ndarray)

    """
    n_features = len(X[1, :])
    beta = cp.Variable(n_features)
    error = cp.sum_squares(X*beta - y)
    obj = cp.Minimize(error + lambda1 * cp.norm(beta, 1) + lambda2*
                      cp.norm(beta[1:n_features]-beta[0:n_features-1], 1))
    prob = cp.Problem(obj)
    prob.solve()

    return beta.value
