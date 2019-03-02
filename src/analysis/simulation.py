import sys
import json
import logging
import pickle
import numpy as np
import cvxpy as cp
from src.model_code.flestimator import FusedLassoEstimator as fle
from src.model_code.functions import fused_lasso_primal
from bld.project_paths import project_paths_join as ppj
from sklearn.model_selection import GridSearchCV
from src.model_code.functions import generate_blocks
import math
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
from time import time

if __name__ == "__main__":
    """Waf."""
    reg_name = sys.argv[1]
    sim_name = sys.argv[2]
    sim_dict = json.load(open(ppj("IN_MODEL_SPECS", sim_name + ".json"),
                         encoding="utf-8"))
    with open(ppj("OUT_ANALYSIS", "data_{}.pickle".
              format(sim_name)), "rb") as in12_file:
        beta_X_epsilon_Y = pickle.load(in12_file)

    """Data import from pickle files."""
    true_beta = beta_X_epsilon_Y[0]                 # p x num_simulations
    X = beta_X_epsilon_Y[1]                         # n x p
    epsilon = beta_X_epsilon_Y[2]                   # n x num_simulations
    y = beta_X_epsilon_Y[3]                         # n x num_simulations

    """Pull Information out of json file."""
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
    num_simulations = sim_dict['num_simulations']  # how many times simulation gets run

    """Building containers to store simulation results."""
    beta_hat = np.empty((p, num_simulations))
    y_hat = np.empty((n, num_simulations))
    residuals = np.empty((n, num_simulations))

    """Calculation of optimal lambda (still missing)."""

    grid_density = 40
    lasso_grid = {
      's1': list(np.linspace(s1_min, s1_max, grid_density))
    }
    fused_grid = {
      's2': list(np.linspace(s2_min, s2_max, grid_density))
    }
    two_d_grid = [{
            's1': list(np.linspace(s1_min, s1_max, grid_density)),
            's2': list(np.linspace(s2_min, s2_max, grid_density))
        }]

    if reg_name == 'lasso':
        lasso_grid = {
            's1': list(np.linspace(s1_min, s1_max, grid_density))
        }
        fused_grid = {
            's2': list(np.linspace(1190, 1200, 1))
        }

        two_d_grid = [{
            's1': list(np.linspace(s1_min, s1_max, grid_density)),
            's2': list(np.linspace(1190, 1200, 5))
        }]

    if reg_name == 'fusion':
        lasso_grid = {
            's1': list(np.linspace(1190, 1200, 1))
        }
        fused_grid = {
            's2': list(np.linspace(s2_min, s2_max, grid_density))
        }
        two_d_grid = [{
            's1': list(np.linspace(1190, 1200, 5)),
            's2': list(np.linspace(s2_min, s2_max, grid_density))
        }]

    start_time_cv = time()

    clf = GridSearchCV(fle(lasso_grid, fused_grid), two_d_grid,
                       scoring='neg_mean_squared_error',
                       n_jobs=-1, iid=False, refit=True,
                       cv=3, verbose=0, pre_dispatch='2*n_jobs',
                       error_score='raise-deprecating',
                       return_train_score='warn')



    clf.fit(X, y[:, 1])
    penalty_cv = [clf.best_params_["s1"], clf.best_params_["s2"]]
    end_time_cv = time()
    # dict of optimal parameters ["s1" : opt_value, "s2": opt_value]
    # np.savetxt("/home/christopher/Dokumente/Testbereich/s1.txt",
    # np.array( [ opt_values["s1"], opt_values["s2"] ] )   )
    # np.savetxt("/home/christopher/Dokumente/
    # Testbereich/beta_hat.txt", beta_hat)

    """Calculation of beta to corresponding optimal lambda."""
    # for sim in range(num_simulations):
    #    beta_hat[:, sim] = fle(opt_values["s1"],opt_values["s2"]).
    # fit( X,y[:, sim])


    for i in range(num_simulations):
        start_time_estimation = time()
        beta_hat[:, i] = fle(penalty_cv[0], penalty_cv[1]).fit(X, y[:, i])
        y_hat[:, i] = np.matmul(X, beta_hat[:, i])
        residuals[:, i] = y[:, i] - y_hat[:, i]
        end_time_estimation = time()



    if reg_name == 'fused':

        time_list = [p, n, np.round((end_time_cv-start_time_cv),2), np.round((end_time_estimation-start_time_estimation),2)]

        with open(ppj("OUT_ANALYSIS", "time_list_{}.pickle".format(sim_name)), "wb") as out_file:
           pickle.dump(time_list, out_file)





    container = [beta_hat, true_beta, penalty_cv, y_hat, residuals, clf.cv_results_['mean_test_score'], clf.cv_results_['params']]
    with open(ppj("OUT_ANALYSIS", "simulation_{}_{}.pickle".format(reg_name, sim_name)), "wb") as out_file:
        pickle.dump(container, out_file)


    # """Analysis of estimator properties and how the true beta got estimated."""
    #
    # # with open("/home/christopher/Dokumente/rm_fused_lasso/bld/out/analysis/simulation_fusion_small_blocks.pickle", "rb") as in12_file:
    # #      analysis = pickle.load(in12_file)
    # # #
    # #  beta_hat = analysis[0]
    # #  true_beta = analysis[1]
    # #
    # container_analysis = []
    # #
    #  # amplitude = 3
    #
    # # number of relevant variables
    # correct_nonzero = sum((beta_hat >= 0.30*amplitude)  & (true_beta > 0)) * 1 / np.sum(true_beta > 0,axis = 0)
    # container_analysis.append(np.mean(correct_nonzero))
    # container_analysis.append(np.std(correct_nonzero))
    #
    #
    #
    # #number of zeros
    # # beta_hat[(np.absolute(beta_hat) <= 0.01) & (true_beta == 0)] = 0
    # # number_correct_zero = np.sum((beta_hat <= 0.1*amplitude) & (true_beta == 0),
    # #                                                             axis=0)
    # #
    # # beta_hat[(np.absolute(beta_hat) <= 0.01) & (true_beta == 0)] = 0
    # percent_correct_zero = np.sum((np.absolute(beta_hat) <= 0.1) & (true_beta == 0),
    #                                                             axis=0) / np.sum(true_beta == 0,axis = 0)
    #
    # container_analysis.append(np.mean(percent_correct_zero))
    # container_analysis.append(np.std(percent_correct_zero))
    #
    #
    #
    #
    #
    # #count the spikes
    # spike_count = []
    # if sim_dict["spikes"] >= 1:
    #     for i in range(num_simulations): #0 bis 199
    #
    #         count = 0
    #
    #         for j in range(p): #0 bis 499
    #
    #             if j == p-1:
    #                 break
    #
    #             if (j == 0) & (true_beta[1,i] == 0) & (beta_hat[j,i] > amplitude/2) & (true_beta[0,i] > 0):
    #
    #                 count = count + 1
    #
    #             if (j == (p-1)) & (beta_hat[(p-1), i] > 2) & (true_beta[p-2, i] == 0) & (true_beta[p-1, i] > 0):
    #                 count = count + 1
    #
    #             if (true_beta[j-1, i] == 0) & (true_beta[j+1, i] == 0) & (beta_hat[j, i] > amplitude/2) & (true_beta[j, i] > 0):
    #
    #                 count = count + 1
    #
    #         spike_count.append(count)
    #     spike_count = np.array(spike_count) / spikes
    #     container_analysis.append(np.mean(spike_count))
    #     container_analysis.append(np.std(spike_count))
    #
    # else:
    #     spike_count = np.zeros(num_simulations) * np.NAN
    #     container_analysis.append(spike_count)
    #     container_analysis.append(spike_count)
    #
    # # count how many blocks got estimated correctly
    # # beta_hat[(beta_hat >= 0.50*amplitude) &
    # #          (beta_hat <= 1.5*amplitude) & (true_beta == amplitude)] = amplitude
    # # beta_hat[(beta_hat >= 0.75*levels) &
    # #          (beta_hat <= 1.25*levels) & (true_beta == levels)] = levels
    # # number_of_blocks = (sum((true_beta == beta_hat) & (beta_hat > 0.5))
    # #                     - spike_count) / length_blocks
    # if reg_name == 'spikes':
    #     number_blocks = 1
    #
    #
    # # if reg_name == 'fusion':
    # #     number_of_blocks = np.sum(((beta_hat >= 0.9*amplitude) & (beta_hat <= 1.1x*amplitude) & (true_beta == amplitude)) | ((beta_hat >= 0.9*levels) & (beta_hat <= 1.1*levels) & (true_beta == levels)),axis = 0)
    # # else:
    # counter_blocks = np.sum(((beta_hat >= 0.50*amplitude) & (beta_hat <= 1.5*amplitude) & (true_beta == amplitude)) | ((beta_hat >= 0.75*levels) & (beta_hat <= 1.25*levels) & (true_beta == levels)),axis = 0)
    #
    # percent_blocks = np.array(counter_blocks) / (length_blocks * number_blocks)
    # container_analysis.append(np.mean(percent_blocks))
    # container_analysis.append(np.std(percent_blocks))
    #
    # with open(ppj("OUT_ANALYSIS", "analysis_{}_{}.pickle".format(reg_name, sim_name)), "wb") as out_file:
    #    pickle.dump(container_analysis, out_file)

    """Plot distribution."""
    beta = true_beta[:, 1]  # nimm ein echtes beta
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

    plt.savefig(ppj("OUT_FIGURES", "plot_{}_{}.png".format(reg_name, sim_name)))

    fig.clf()
