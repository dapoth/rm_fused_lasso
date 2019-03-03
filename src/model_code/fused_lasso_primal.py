import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pandas as pd
import random as rd
from sklearn import model_selection
import math

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
