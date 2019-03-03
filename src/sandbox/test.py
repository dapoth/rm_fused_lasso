#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 23:49:37 2019

@author: clara
"""





import numpy as np
import pickle
import cvxpy as cp

def fused_lasso_dual(y, x, lambda1, lambda2):
    """
    Solves for given data and penalty constants the fused lasso dual form.
    """
    p = len(x[1, :])
    b = cp.Variable(p)
    error = cp.sum_squares(x*b - y)
    obj = cp.Minimize(error + lambda1 * cp.norm(b, 1) +
                      lambda2 * cp.norm(b[1:p]-b[0:p-1], 1))
    prob = cp.Problem(obj)
    prob.solve()

    return b.value

def fused_lasso_primal(y, x, s1, s2):

    """Docstring."""

    # "from constraint import un_constraint" to import function
    # y and x data as usual

    p = len(x[1, :])
    b = cp.Variable(p)
    error = cp.sum_squares(x*b - y)
    obj = cp.Minimize(error)
    constraints = [cp.norm(b, 1) <= s1, cp.norm(b[1:p]-b[0:p-1], 1) <= s2]
    prob = cp.Problem(obj, constraints)
    prob.solve()

    return b.value

def calc_lambda(y, X, s1, beta_s):

    lambda1 = 0
    beta_l = fused_lasso_dual(y, X, lambda1, 0)

    while np.sum(beta_l-beta_s) > 0.0000001:

        lambda1 = lambda1 + 0.05

        beta_l = fused_lasso_dual(y, X, lambda1, 0)

        print(np.sum(beta_l-beta_s))

        if lambda1 > 100:
            break

    return lambda1


with open("/home/christopher/Dokumente/rm_fused_lasso/bld/out/data/data_spikes.pickle", "rb") as in12_file:
    beta_X_epsilon_Y = pickle.load(in12_file)

X = beta_X_epsilon_Y[1]
y = beta_X_epsilon_Y[3][:,1]

s1 = 140

init = fused_lasso_primal(y,X,s1,0)

test = calc_lambda(y,X,140,init)
