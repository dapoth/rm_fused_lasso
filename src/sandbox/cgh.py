#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 23:49:37 2019

@author: clara
"""





import matplotlib.pyplot as plt
import cvxpy as cp
import numpy as np



def fused_lasso_primal(y,x,s1,s2):

    ### "from constraint import un_constraint" to import function
    ### y and x data as usual

    p = len(x[1,:])
    b = cp.Variable(p)
    error = cp.sum_squares(x*b - y)
    obj = cp.Minimize(error)
    constraints = [cp.norm(b,1) <= s1, cp.norm(b[1:p]-b[0:p-1],1) <= s2]
    prob = cp.Problem(obj, constraints)
    prob.solve()

    return b.value



# Plot cgh_data

cgh_data = np.loadtxt("cgh.txt")
beta = fused_lasso_primal(cgh_data, np.identity(len(cgh_data)), 160, 15)
plt.xlabel('Genome Order')
plt.ylabel('Copy Number')
plt.plot(cgh_data,"bo")
plt.axhline(color='r')
plt.plot(beta, color='orange')
plt.savefig("cgh_plot.png")
plt.show()
