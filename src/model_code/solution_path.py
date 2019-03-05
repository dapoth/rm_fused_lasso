import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt


def solution_path_unconstraint(y, X, lambda1=0, lambda2=0):

    ### "from constraint import constraint" to import function
    ### y and x data as usual
    ### lambda1 and lambda2 optional to make vertical line in the plot

    n_features = len(X[1, :])
    gamma1 = cp.Parameter(nonneg=True)
    gamma2 = cp.Parameter(nonneg=True)
    beta_hat = cp.Variable(n_features)
    error = cp.sum_squares(X*beta_hat - y)
    obj = cp.Minimize(error+gamma1*cp.norm(beta_hat, 1)
                      +gamma2*cp.norm(beta_hat[1:n_features]-beta_hat[0:n_features-1], 1))
    prob = cp.Problem(obj)


    x_values = []
    gamma_vals = np.logspace(-2, 6)
    for val in gamma_vals:
        gamma1.value = val
        gamma2.value = lambda2
        prob.solve()
        x_values.append(beta_hat.value)

    x2_values = []
    gamma2_vals = np.logspace(-2, 6)
    for val in gamma_vals:
        gamma1.value = lambda1
        gamma2.value = val
        prob.solve()
        x2_values.append(beta_hat.value)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=(6, 10))

    # Plot entries of x vs. lambda1.
    plt.subplot(211)
    plt.axvline(X=lambda1)
    for i in range(n_features):
        plt.plot(gamma_vals, [xi[i] for xi in x_values])
    plt.xlabel(r'$\lambda_1$', fontsize=16)
    plt.ylabel(r'$\beta_{i}$', fontsize=16)
    plt.xscale('log')
    plt.title(r'Entries of $\beta$ vs. $\lambda_1$')

    plt.subplot(212)
    plt.axvline(X=lambda2)
    for i in range(n_features):
        plt.plot(gamma2_vals, [xi[i] for xi in x2_values])
    plt.xlabel(r'$\lambda_2$', fontsize=16)
    plt.ylabel(r'$\beta_{i}$', fontsize=16)
    plt.xscale('log')
    plt.title(r'Entries of $\beta$ vs. $\lambda_2$')

    plt.tight_layout()
    plt.show()

    return print("The prcoess was", prob.status)
