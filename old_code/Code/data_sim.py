import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from functions import solution_path_lasso_signal
import random as rm

np.random.seed(1)
#Parameters
n = 100
p = 100
positive_betas = 10
number_of_blocks = 3
amplitude=1
p_scale = np.arange(1, p+1,1)



# generate design matrix
mean = np.ones(p)
cov = np.identity(p)
X = np.random.multivariate_normal(mean, cov,n)
I = np.identity(n) # creates an identity matrix of size n

# draw error term
eps = np.random.randn(n)

#generate beta
#beta = np.random.randn(p) * 4
beta = np.zeros(p)
index_for_data = rm.sample(range(0,p),number_of_blocks)

beta[index_for_data]=amplitude
print("Oracle Regressoren:",beta)
#generate y
#y = np.random.randn(10)
y = np.dot(X,beta) + eps

# load real application data
cgh_data = np.loadtxt("cgh.txt", delimiter=',')
