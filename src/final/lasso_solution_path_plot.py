import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pandas as pd
import random as rd
from sklearn import model_selection
import math
from bld.project_paths import project_paths_join as ppj

def solution_path_unconstraint(y,x):

    ### "from constraint import constraint" to import function
    ### y and x data as usual
    ### lambda1 and lambda2 optional to make vertical line in the plot


    p = len(x[1,:])
    gamma1 = cp.Parameter(nonneg=True)
    b = cp.Variable(p)
    error = cp.sum_squares(y-x*b)
    obj = cp.Minimize(error+gamma1*cp.norm(b,1))
    prob = cp.Problem(obj)


    x_values = []
    gamma_vals = np.linspace(0, 75)
    for val in gamma_vals:
        gamma1.value = val
        prob.solve()
        x_values.append(b.value)



    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Plot entries of x vs. lambda1.

    for i in range(p):
        plt.plot(gamma_vals, [xi[i] for xi in x_values])
    plt.xlabel(r'$\lambda_1$', fontsize=16)
    plt.ylabel(r'$\hat{\beta}_{i}$', fontsize=16)
    plt.xscale('log')
    plt.title(r'Entries of $\hat{\beta}$ vs. $\lambda_1$')





    plt.savefig(ppj("OUT_FIGURES", "plot_solutionpath_lasso.png"))





    #prob.solve()

    return




p = 10
n = 10
betas = np.zeros(p)
betas[3:6] = 1


np.random.seed(1000)
mean = np.zeros(p)
cov = np.identity(p)
X = np.random.multivariate_normal(mean, cov, n)
eps = np.random.randn(n)
Y = np.matmul(X, betas) + eps

solution_path_unconstraint(Y,X)


betas




# n = 15
# m = 10
# np.random.seed(1)
# A = np.random.randn(n, m)
# b = np.random.randn(n)
# # gamma must be nonnegative due to DCP rules.
# gamma = cp.Parameter(nonneg=True)
#
# # Construct the problem.
# x = cp.Variable(m)
# error = cp.sum_squares(A*x - b)
# obj = cp.Minimize(error + gamma*cp.norm(x, 1))
# prob = cp.Problem(obj)
#
# # Construct a trade-off curve of ||Ax-b||^2 vs. ||x||_1
# sq_penalty = []
# l1_penalty = []
# x_values = []
# gamma_vals = np.logspace(-4, 6)
# for val in gamma_vals:
#     gamma.value = val
#     prob.solve()
#     # Use expr.value to get the numerical value of
#     # an expression in the problem.
#     sq_penalty.append(error.value)
#     l1_penalty.append(cp.norm(x, 1).value)
#     x_values.append(x.value)
#
#
#
#
#
# # Plot entries of x vs. gamma.
# plt.subplot(212)
# for i in range(m):
#     plt.plot(gamma_vals, [xi[i] for xi in x_values])
#
#
# plt.tight_layout()
# plt.show()
