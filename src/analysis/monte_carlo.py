import sys
import json
import pickle
import numpy as np
from src.model_code.fused_lasso_primal import fused_lasso_primal
from sklearn.model_selection import GridSearchCV
from bld.project_paths import project_paths_join as ppj
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """ waf """

    with open(ppj("OUT_ANALYSIS", "simulation_fused_large_blocks.pickle"), "rb") as in12_file:
                  analysis = pickle.load(in12_file)

    sim_dict = json.load(open(ppj("IN_MODEL_SPECS", "large_blocks.json"),
                             encoding="utf-8"))


    """Plot distribution."""
    beta = analysis[1][:,1]   #true_beta[:, 1]  # nimm ein echtes beta
    penalty_cv = analysis[2]



    p = sim_dict["p"]
    n = sim_dict["n"]
    s1_min = sim_dict["s1_min"]
    s1_max = sim_dict["s1_max"]
    s2_min = sim_dict["s2_min"]
    s2_max = sim_dict["s2_max"]
    num_simulations = sim_dict['num_simulations']

    beta_container = np.ones((p, num_simulations))
    mean_x = np.zeros(p)
    cov_X = np.identity(p)
    X = np.random.multivariate_normal(mean_x, cov_X, n)

    for j in range(num_simulations):

        eps = np.random.rand(n)
        y = np.matmul(X, beta) + eps
        beta_container[:, j] = fused_lasso_primal(y, X, penalty_cv[0], penalty_cv[1])



    """Plot distribution of beta_j before block at break of block and inside block"""
    list_index = []
    for ch in range(p):
        if ch == (p-4):
            list_index = [4, 7, 9, 12]
            break

        if (beta[ch] == 0) & (beta[(ch+1)] == 0) & (beta[(ch+2)] != 0) & (beta[(ch+3)] != 0):
            list_index = [ch+3, ch, ch+2, ch+1]
            break

    fig, axes = plt.subplots(2, 2)

    axes[0, 0].set_title('center')
    axes[0, 0].hist(beta_container[list_index[0], :])

    axes[1, 0].set_xlabel('zero')
    axes[1, 0].hist(beta_container[list_index[1], :])

    axes[0, 1].hist(beta_container[list_index[2], :])
    axes[0, 1].set_title('block_in')

    axes[1, 1].hist(beta_container[list_index[3], :])
    axes[1, 1].set_xlabel('block_out')

    plt.savefig(ppj("OUT_FIGURES", "monte_carlo_fused_large_blocks.png"))
