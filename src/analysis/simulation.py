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
    grid_density = sim_dict['grid_density']
    num_simulations = sim_dict['num_simulations']  # how many times simulation gets run

    """Building containers to store simulation results."""
    beta_hat = np.empty((p, num_simulations))
    y_hat = np.empty((n, num_simulations))
    residuals = np.empty((n, num_simulations))

    """Calculation of optimal lambda (still missing)."""


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
