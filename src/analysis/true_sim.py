import sys
import json
import logging
import pickle
import numpy as np
import cvxpy as cp

from src.model_code.agent import Agent
from bld.project_paths import project_paths_join as ppj




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
    plt.figure(figsize=(6,10))

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






if __name__ == "__main__":
    sim_name = sys.argv[1]
    model = json.load(open(ppj("IN_MODEL_SPECS", sim_name + ".json"), encoding="utf-8"))

    sim_dict = json.load(open(ppj("IN_MODEL_SPECS", sim_name + ".json"), encoding="utf-8"))
    #logging.basicConfig(
    #    filename=ppj("OUT_ANALYSIS", "log", "simulation_{}.log".format(model_name)),
    #    filemode="w",
    #    level=logging.INFO
    #)


    #np.random.seed(1) #model["rng_seed"]
    #logging.info(model['penalty1']) #"rng_seed"


    with open(ppj("OUT_ANALYSIS", "simulation_{}.pickle".format(sim_name)), "rb") as in_file:
        data = pickle.load(in_file)

    beta = data[0]
    beta_hat = data[1]
    X =data[2]
    X_T = data[3]
    epsilion = data[4]
    y = data[5]
    num_simulations = sim_dict['num_simulations']
    s_1 = sim_dict['s1']
    s_2 = sim_dict['s2']

    for sim in range(num_simulations):
        beta_hat[:,sim] = fused_lasso_primal(y[:,sim],X, s_1, s_2)

    y_hat = np.matmul(X, beta_hat)
    epsilon_hat = y - y_hat
