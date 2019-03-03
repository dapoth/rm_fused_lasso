import sys
import json
import logging
import pickle
import numpy as np
import cvxpy as cp
from src.model_code.flestimator import FusedLassoEstimator as fle
from src.model_code.fused_lasso_primal import fused_lasso_primal
from sklearn.model_selection import GridSearchCV
from src.model_code.generate_blocks import generate_blocks
import math
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
from time import time
from bld.project_paths import project_paths_join as ppj




if __name__ == "__main__":
    """ waf """
    sim_name = sys.argv[2]
    reg_name = sys.argv[1]
    with open(ppj("OUT_ANALYSIS", "simulation_{}_{}.pickle".
                  format(reg_name, sim_name)), "rb") as in12_file:
                  analysis = pickle.load(in12_file)

    sim_dict = json.load(open(ppj("IN_MODEL_SPECS", sim_name + ".json"),
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
    number_blocks = sim_dict['number_of_blocks']
    length_blocks = sim_dict['length_blocks']
    amplitude = sim_dict['amplitude']
    spike_level = sim_dict['spike_level']
    levels = sim_dict['levels']
    spikes = sim_dict['spikes']
    num_simulations = 200    #sim_dict['num_simulations']

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

    plt.savefig(ppj("OUT_FIGURES", "monte_carlo_{}_{}.png".format(reg_name, sim_name)))
