"""Bla."""
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from bld.project_paths import project_paths_join as ppj

def solution_path_unconstraint(response, explanatory_var):
    """Calculate and plot solution path of fused lasso for fixed lasso penatlyself.

    Args:
        response (np.ndarray): 1d array of responses
        explanatory_var (np.ndarray): 2d array of explanatory variables

    Returns:

    """
    n_features = len(explanatory_var[1, :])

    #Set up fused lasso problem
    gamma1 = cp.Parameter(nonneg=True)
    gamma2 = cp.Parameter(nonneg=True)
    beta_hat = cp.Variable(n_features)
    error = cp.sum_squares(explanatory_var*beta_hat - response)
    obj = cp.Minimize(error + gamma1*cp.norm(beta_hat, 1)
                      + gamma2*cp.norm(beta_hat[1:n_features]-beta_hat[0:n_features-1], 1))
    prob = cp.Problem(obj)

    #Set up grid for lambda2 term and solve fused lasso problem wit CVXPY
    x2_values = []
    gamma2_vals = np.linspace(0, 100)
    for val in gamma2_vals:
        gamma1.value = 1
        gamma2.value = val
        prob.solve()
        x2_values.append(beta_hat.value)

    # Plot entries of x vs. lambda1.
    for i in range(n_features):
        plt.plot(gamma2_vals, [xi[i] for xi in x2_values])
    plt.xlabel(r'$\lambda_2$', fontsize=16)
    plt.ylabel(r'$\hat{\beta}_{i}$', fontsize=16)
    plt.title('Neighbouring to the block')

    plt.savefig(ppj("OUT_FIGURES", "plot_solutionpath_fused_lasso.png"))



N_FEATURES = 10
N_OBS = 10
BETA_HAT = np.zeros(N_FEATURES)
BETA_HAT[3:6] = 3

np.random.seed(1000)
MEAN = np.zeros(N_FEATURES)
COV = np.identity(N_FEATURES)
DESIGN_MAT = np.random.multivariate_normal(MEAN, COV, N_OBS)
EPS = np.random.randn(N_OBS)
OUTCOME = np.matmul(DESIGN_MAT, BETA_HAT) + EPS

solution_path_unconstraint(OUTCOME, DESIGN_MAT)
