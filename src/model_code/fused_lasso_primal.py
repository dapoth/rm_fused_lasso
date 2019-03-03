import cvxpy as cp

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
    # "from constraint import un_constraint" to import function
    # y and x data as usual

    n_features = len(X[1, :])
    beta = cp.Variable(n_features)
    error = cp.sum_squares(X* beta - y)
    obj = cp.Minimize(error)
    constraints = ([cp.norm(beta, 1) <= s1,
                    cp.norm(beta[1:n_features]-beta[0:n_features-1], 1) <= s2])
    prob = cp.Problem(obj, constraints)
    prob.solve()

    return beta.value
