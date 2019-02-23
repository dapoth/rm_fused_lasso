import pickle
import numpy as np
import math
import random as rd
import matplotlib.pyplot as plt
from src.model_code.functions import fused_lasso_dual
from bld.project_paths import project_paths_join as ppj
import cvxpy as cp


def fused_lasso_dual(y,x,lambda1,lambda2):

    p = len(x[1,:])
    b = cp.Variable(p)
    error = cp.sum_squares(x*b - y)
    obj = cp.Minimize(error+lambda1*cp.norm(b,1)  +lambda2*cp.norm(b[1:p]-b[0:p-1],1))
    prob = cp.Problem(obj)
    prob.solve()

    return b.value


p = 300
n = 100
betas = np.zeros(300)

betas[10:20] = 2
betas[50:60] = 1
betas[200:230] = 1.5
betas[240:260] = 1
betas[280:290] = 1

mean = np.zeros(p)
cov = np.identity(p)
X = np.random.multivariate_normal(mean, cov, n)
eps = np.random.randn(n)
Y = np.matmul(X, betas) + eps

beta_good = fused_lasso_dual(Y,X,75,250)

beta_fused = fused_lasso_dual(Y,X,10,250)

beta_lasso = fused_lasso_dual(Y,X,250,10)

beta_both = fused_lasso_dual(Y,X,250,250)


fig, axes = plt.subplots(2, 2, gridspec_kw = {"height_ratios": [1,1]})


axes[0,0].set_title('Good Penaltys')
axes[0, 0].plot(betas)
axes[0, 0].plot(beta_good)

axes[1,0].set_xlabel('High fsusion')
axes[1, 0].plot(betas)
axes[1, 0].plot(beta_fused)

axes[0, 1].plot(betas)
axes[0, 1].plot(beta_lasso)
axes[0,1].set_title('High Lasso')


axes[1, 1].plot(betas)
axes[1, 1].plot(beta_both)
axes[1,1].set_xlabel('Both High')




plt.savefig(ppj("OUT_FIGURES", "different_penaltys.pdf"))
