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
    gamma2_vals = np.logspace(-2,6)
    for val in gamma2_vals:
        gamma1.value = 1
        gamma2.value = val
        prob.solve()
        x2_values.append(b.value)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=(6, 10))

    # Plot entries of x vs. lambda1.


    plt.subplot(212)
    for i in range(p):
        plt.plot(gamma2_vals, [xi[i] for xi in x2_values])
    plt.xlabel(r'$\lambda_2$', fontsize=16)
    plt.ylabel(r'$\beta_{i}$', fontsize=16)
    plt.xscale('log')
    plt.title(r'Entries of $\beta$ vs. $\lambda_2$')


    plt.savefig(ppj("OUT_FIGURES", "plot_solutionpath_fused_lasso.png"))




    #prob.solve()

    return




p = 20
n = 10
betas = np.ones(20)
betas[1:3] = 3
betas[5:7] = 2
betas[15:19] = -1


np.random.seed(1000)
mean = np.zeros(p)
cov = np.identity(p)
X = np.random.multivariate_normal(mean, cov, n)
eps = np.random.randn(n)
Y = np.matmul(X, betas) + eps

solution_path_unconstraint(Y,X)
