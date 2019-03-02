import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pandas as pd
import random as rd
from sklearn import model_selection
import math


def solution_path_unconstraint(y,x,lambda1=0,lambda2=0):

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


    x_values = []
    gamma_vals = np.logspace(-2, 6)
    for val in gamma_vals:
        gamma1.value = val
        gamma2.value = lambda2
        prob.solve()
        x_values.append(b.value)

    x2_values = []
    gamma2_vals = np.logspace(-2,6)
    for val in gamma_vals:
        gamma1.value = lambda1
        gamma2.value = val
        prob.solve()
        x2_values.append(b.value)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=(6, 10))

    # Plot entries of x vs. lambda1.
    plt.subplot(211)
    plt.axvline(x=lambda1)
    for i in range(p):
        plt.plot(gamma_vals, [xi[i] for xi in x_values])
    plt.xlabel(r'$\lambda_1$', fontsize=16)
    plt.ylabel(r'$\beta_{i}$', fontsize=16)
    plt.xscale('log')
    plt.title(r'Entries of $\beta$ vs. $\lambda_1$')

    plt.subplot(212)
    plt.axvline(x=lambda2)
    for i in range(p):
        plt.plot(gamma2_vals, [xi[i] for xi in x2_values])
    plt.xlabel(r'$\lambda_2$', fontsize=16)
    plt.ylabel(r'$\beta_{i}$', fontsize=16)
    plt.xscale('log')
    plt.title(r'Entries of $\beta$ vs. $\lambda_2$')

    plt.tight_layout()
    plt.show()




    #prob.solve()

    return print("The prcoess was",prob.status)
