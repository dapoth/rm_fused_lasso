import cvxpy as cp
import numpy as np
from src.model_code.fused_lasso_duall import fused_lasso_dual


def calc_lambda(y, X, s1, beta_s):

    lambda1 = 0
    beta_l = fused_lasso_dual(y, X, lambda1, 0)

    while np.sum(np.square(beta_l-beta_s)) > 0.001:

        lambda1 = lambda1 + 0.01

        beta_l = fused_lasso_dual(y, X, lambda1, 0)

        if lambda1 == 10:
            break

    return lambda1
