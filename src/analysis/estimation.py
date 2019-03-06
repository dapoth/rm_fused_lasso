import sys
import json
import logging
import pickle
import numpy as np
from src.model_code.flestimator import FusedLassoEstimator as fle
from src.model_code.fused_lasso_primal import fused_lasso_primal
from bld.project_paths import project_paths_join as ppj
from sklearn.model_selection import GridSearchCV
from time import time

if __name__ == "__main__":
    """Waf."""
    reg_name = sys.argv[1]
    sim_name = sys.argv[2]
    sim_dict = json.load(open(ppj("IN_MODEL_SPECS", sim_name + ".json"),
                         encoding="utf-8"))
    with open(ppj("OUT_DATA", "data_{}.pickle".
              format(sim_name)), "rb") as in_file:
        beta_X_epsilon_Y = pickle.load(in_file)

    with open(ppj("OUT_ANALYSIS", "cv_{}_{}.pickle".
              format(reg_name, sim_name)), "rb") as in_file:
        cv = pickle.load(in_file)

    """Data import from pickle files."""
    true_beta = beta_X_epsilon_Y[0]                 # p x num_simulations
    X = beta_X_epsilon_Y[1]                         # n x p
    epsilon = beta_X_epsilon_Y[2]                   # n x num_simulations
    y = beta_X_epsilon_Y[3]                         # n x num_simulations

    """Data import from pickle files."""
    s1_opt = cv[2][0]
    s2_opt = cv[2][1]

    """Pull Information out of json file."""
    p = sim_dict["p"]
    n = sim_dict["n"]
    s1_min = sim_dict["s1_min"]
    s1_max = sim_dict["s1_max"]
    s2_min = sim_dict["s2_min"]
    s2_max = sim_dict["s2_max"]
    grid_density = sim_dict['grid_density']
    num_simulations = sim_dict['num_simulations']  # how many times simulation gets run

    """Building containers to store simulation results."""
    beta_hat = np.empty((p, num_simulations))
    y_hat = np.empty((n, num_simulations))
    residuals = np.empty((n, num_simulations))




    """Calculation of beta to corresponding optimal lambda."""



    for i in range(num_simulations):
        start_time_estimation = time()
        beta_hat[:, i] = fused_lasso_primal(y[:, i], X, s1_opt, s2_opt) #fle(penalty_cv[0], penalty_cv[1]).fit(X, y[:, i])
        y_hat[:, i] = np.matmul(X, beta_hat[:, i])
        residuals[:, i] = y[:, i] - y_hat[:, i]
        end_time_estimation = time()



    if reg_name == 'fused':

        with open(ppj("OUT_ANALYSIS", "time_list_{}.pickle".format(sim_name)), "rb") as in_file:
            time_list = pickle.load(in_file)

        time_estimation = np.round((end_time_estimation-start_time_estimation), 2)
        time_list.append(time_estimation)

        with open(ppj("OUT_ANALYSIS", "time_list_{}.pickle".format(sim_name)), "wb") as out_file:
           pickle.dump(time_list, out_file)


    penalty_cv = [s1_opt, s2_opt]


    container = [beta_hat, true_beta, penalty_cv, y_hat, residuals, cv[0], cv[1]]
    with open(ppj("OUT_ANALYSIS", "simulation_{}_{}.pickle".format(reg_name, sim_name)), "wb") as out_file:
        pickle.dump(container, out_file)
