"""Perform Monte Carlo simulation for the large_blocks fused lasso setting.

Repeated estimation of specific coefficients and plotting histograms of estimated
coefficients.

"""
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from src.model_code.fused_lasso_primal import fused_lasso_primal
from bld.project_paths import project_paths_join as ppj

if __name__ == "__main__":

    with open(ppj("OUT_ANALYSIS", "simulation_fused_large_blocks.pickle"), "rb") as in_file:
        ANALYSIS = pickle.load(in_file)

    SIM_DICT = json.load(open(ppj("IN_MODEL_SPECS", "large_blocks.json"),
                              encoding="utf-8"))

    # Load data from pickle file.
    BETA = ANALYSIS[1][:, 1]
    S1_OPT = ANALYSIS[2][0]
    S2_OPT = ANALYSIS[2][1]

    # Load data from json file.
    N_FEATURES = SIM_DICT["p"]
    N_OBS = SIM_DICT["n"]
    N_SIMULATIONS = SIM_DICT['num_simulations']

    # Initialize simulation data.
    MEAN_X = np.zeros(N_FEATURES)
    COV_X = np.identity(N_FEATURES)
    X = np.random.multivariate_normal(MEAN_X, COV_X, N_OBS)

    # Estimate beta coefficients for each simulation step.
    BETA_HAT_CONTAINER = np.ones((N_FEATURES, N_SIMULATIONS))
    for j in range(N_SIMULATIONS):
        eps = np.random.rand(N_OBS)
        y = np.matmul(X, BETA) + eps
        BETA_HAT_CONTAINER[:, j] = fused_lasso_primal(y, X, S1_OPT, S2_OPT)

    # Plot distribution of specific estimated beta_j:
    # Distant to block, before block, at block border and inside block.
    LIST_INDEX = []
    for i in range(N_FEATURES):
        if i == (N_FEATURES-4):
            LIST_INDEX = [4, 7, 9, 12]
            break

        if (BETA[i] == 0) & (BETA[(i+1)] == 0) & (BETA[(i+2)] != 0) & (BETA[(i+3)] != 0):
            LIST_INDEX = [i+3, i, i+2, i+1]
            break

    FIG, AXES = plt.subplots(2, 2)

    AXES[0, 0].set_title('center')
    AXES[0, 0].hist(BETA_HAT_CONTAINER[LIST_INDEX[0], :])

    AXES[1, 0].set_xlabel('zero')
    AXES[1, 0].hist(BETA_HAT_CONTAINER[LIST_INDEX[1], :])

    AXES[0, 1].hist(BETA_HAT_CONTAINER[LIST_INDEX[2], :])
    AXES[0, 1].set_title('block_in')

    AXES[1, 1].hist(BETA_HAT_CONTAINER[LIST_INDEX[3], :])
    AXES[1, 1].set_xlabel('block_out')

    plt.savefig(ppj("OUT_FIGURES", "monte_carlo_fused_large_blocks.png"))
    plt.clf()
