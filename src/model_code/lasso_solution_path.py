import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from bld.project_paths import project_paths_join as ppj


def lasso_solution_path(y,x):

    ### "from constraint import constraint" to import function
    ### y and x data as usual
    ### lambda1 and lambda2 optional to make vertical line in the plot


    p = len(x[1,:])
    gamma1 = cp.Parameter(nonneg=True)
    b = cp.Variable(p)
    error = cp.sum_squares(y-x*b)
    obj = cp.Minimize(error+gamma1*cp.norm(b,1))
    prob = cp.Problem(obj)


    x_values = []
    gamma_vals = np.linspace(0, 75)
    for val in gamma_vals:
        gamma1.value = val
        prob.solve()
        x_values.append(b.value)



    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Plot entries of x vs. lambda1.

    for i in range(p):
        plt.plot(gamma_vals, [xi[i] for xi in x_values])
    plt.xlabel(r'$\lambda_1$', fontsize=16)
    plt.ylabel(r'$\hat{\beta}_{i}$', fontsize=16)
    plt.xscale('log')
    plt.title(r'Entries of $\hat{\beta}$ vs. $\lambda_1$')





    plt.savefig(ppj("OUT_FIGURES", "plot_solutionpath_lasso.png"))





    #prob.solve()

    return
