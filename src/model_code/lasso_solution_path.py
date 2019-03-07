import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
#from bld.project_paths import project_paths_join as ppj


def lasso_solution_path(y,X):
    """Calculate and plot solution path for lasso.

    Args:
        | y (np.ndarray): 1d array of responses
        | X (np.ndarray): 2d array of explanatory variables

    Returns:
        | plt (matplotlib.pyplot)

    """
    # Set up lasso problem.
    n_features = len(X[1, :])
    gamma1 = cp.Parameter(nonneg=True)
    beta_hat = cp.Variable(n_features)
    error = cp.sum_squares(y - X*beta_hat)
    obj = cp.Minimize(error + gamma1*cp.norm(beta_hat, 1))
    prob = cp.Problem(obj)

    x_values = []
    gamma_vals = np.linspace(0, 75)
    for val in gamma_vals:
        gamma1.value = val
        prob.solve()
        x_values.append(beta_hat.value)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Plot entries of estimated beta vs. lambda1.
    for i in range(n_features):
        plt.plot(gamma_vals, [xi[i] for xi in x_values])
    plt.xlabel(r'$\lambda_1$', fontsize=16)
    plt.ylabel(r'$\hat{\beta}_{i}$', fontsize=16)
    plt.xscale('log')
    plt.title(r'Entries of $\hat{\beta}$ vs. $\lambda_1$')

    return plt
