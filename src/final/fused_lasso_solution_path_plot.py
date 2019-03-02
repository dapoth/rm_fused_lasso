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
    gamma2 = cp.Parameter(nonneg=True)
    b = cp.Variable(p)
    error = cp.sum_squares(x*b - y)
    obj = cp.Minimize(error+gamma1*cp.norm(b,1)  +gamma2*cp.norm(b[1:p]-b[0:p-1],1))
    prob = cp.Problem(obj)




    x2_values = []
    gamma2_vals = np.linspace(0,100)
    for val in gamma2_vals:
        gamma1.value = 1
        gamma2.value = val
        prob.solve()
        x2_values.append(b.value)



    # Plot entries of x vs. lambda1.


    for i in range(p):
        plt.plot(gamma2_vals, [xi[i] for xi in x2_values])
    plt.xlabel(r'$\lambda_2$', fontsize=16)
    plt.ylabel(r'$\hat{\beta}_{i}$', fontsize=16)

    plt.title('Neighbouring to the block')


    plt.savefig(ppj("OUT_FIGURES", "plot_solutionpath_fused_lasso.png"))

    return




p = 10
n = 10
betas = np.zeros(p)
betas[3:6] = 3


np.random.seed(1000)
mean = np.zeros(p)
cov = np.identity(p)
X = np.random.multivariate_normal(mean, cov, n)
eps = np.random.randn(n)
Y = np.matmul(X, betas) + eps

solution_path_unconstraint(Y,X)
