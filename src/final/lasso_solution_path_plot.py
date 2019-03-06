import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from src.model_code.lasso_solution_path import lasso_solution_path

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

lasso_solution_path(Y, X)


betas

n = 15
m = 10
np.random.seed(1)
A = np.random.randn(n, m)
b = np.random.randn(n)
# gamma must be nonnegative due to DCP rules.
gamma = cp.Parameter(nonneg=True)

# Construct the problem.
x = cp.Variable(m)
error = cp.sum_squares(A*x - b)
obj = cp.Minimize(error + gamma*cp.norm(x, 1))
prob = cp.Problem(obj)

# Construct a trade-off curve of ||Ax-b||^2 vs. ||x||_1
sq_penalty = []
l1_penalty = []
x_values = []
gamma_vals = np.logspace(-4, 6)
for val in gamma_vals:
    gamma.value = val
    prob.solve()
    # Use expr.value to get the numerical value of
    # an expression in the problem.
    sq_penalty.append(error.value)
    l1_penalty.append(cp.norm(x, 1).value)
    x_values.append(x.value)





# Plot entries of x vs. gamma.
plt.subplot(212)
for i in range(m):
    plt.plot(gamma_vals, [xi[i] for xi in x_values])


plt.tight_layout()
