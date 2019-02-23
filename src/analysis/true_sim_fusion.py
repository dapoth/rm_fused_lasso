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



if __name__ == "__main__":

    """ waf """
    sim_name = sys.argv[1]
    sim_dict = json.load(open(ppj("IN_MODEL_SPECS", sim_name + ".json"), encoding="utf-8"))
    with open(ppj("OUT_ANALYSIS", "data_simulation_{}.pickle".format(sim_name)), "rb") as in12_file:
        beta_X_epsilon_Y = pickle.load(in12_file)



    """data import from pickle files"""
    true_beta = beta_X_epsilon_Y[0]          #beta is p x num_simulations
      #beta is p x num_simulations
    X = beta_X_epsilon_Y[1]             # n x p
    epsilon = beta_X_epsilon_Y[2]      # n x num_simulations
    y = beta_X_epsilon_Y[3]             # n x num_simulations
    num_simulations = sim_dict['num_simulations'] # how many times simulation gets run
    p = sim_dict["p"]
    n = sim_dict["n"]

    """building containers to store simulation results """

    beta_hat = np.empty((p,num_simulations))
    y_hat = np.empty((n,num_simulations))
    residuals = np.empty((n,num_simulations))


    """Calculation of optimal lambda (still missing)"""

    #s_1 = sim_dict['s1']
    #s_2 = sim_dict['s2']

    lasso_grid = {
      's1': list(np.linspace(1000,1200,5))
    }
    fused_grid = {
      's2': list(np.linspace(sim_dict["s2_min"],sim_dict["s2_max"],1))
    }

    two_d_grid = [{
                's1': list(np.linspace(1000,1200,5)),
                's2': list(np.linspace(sim_dict["s2_min"],sim_dict["s2_max"],50))
                }]

    clf = GridSearchCV(fle(lasso_grid,fused_grid), two_d_grid,
                            scoring= 'neg_mean_squared_error',
                            n_jobs = -1, iid= False, refit=True,
                            cv=None, verbose=0, pre_dispatch='2*n_jobs',
                            error_score='raise-deprecating',
                            return_train_score='warn')






    clf.fit(X, y[:,1])
    penalty_cv = [clf.best_params_["s1"], clf.best_params_["s2"]]
     # dict of optimal parameters ["s1" : opt_value, "s2": opt_value]

    #np.savetxt("/home/christopher/Dokumente/Testbereich/s1.txt", np.array( [ opt_values["s1"], opt_values["s2"] ] )   )
    #np.savetxt("/home/christopher/Dokumente/Testbereich/beta_hat.txt", beta_hat)


    """calculation of beta to corresponding optimal lambda"""
    #for sim in range(num_simulations):
    #    beta_hat[:, sim] = fle(opt_values["s1"],opt_values["s2"]).fit( X,y[:, sim])

    for i in range(num_simulations):
        beta_hat[:,i] = fle(penalty_cv[0], penalty_cv[1]).fit(X, y[:,i])
        y_hat[:,i] = np.matmul(X,beta_hat[:,i])
        residuals[:,i] = y[:,i] - y_hat[:,i]

    container = [beta_hat, true_beta, penalty_cv, y_hat, residuals]
    with open(ppj("OUT_ANALYSIS", "simulated_beta_hat_beta_penalty_cv_y_hat_residuals_fusion{}.pickle".format(sim_name)), "wb") as out_file:
        pickle.dump(container, out_file)

    # # Load Data from pickle file
    # beta_hat = simulated_beta_hat_beta_penalty_cv_y_hat_residuals[0]
    # beta = simulated_beta_hat_beta_penalty_cv_y_hat_residuals[1]
    # penalty_cv =simulated_beta_hat_beta_penalty_cv_y_hat_residuals[2]
    # y_hat = simulated_beta_hat_beta_penalty_cv_y_hat_residuals[3]
    # residuals = simulated_beta_hat_beta_penalty_cv_y_hat_residuals[4]

    # Load data from json
    amplitude = sim_dict["amplitude"]
    num_simulations = sim_dict["num_simulations"]
    num_of_blocks = sim_dict["number_of_blocks"]
    length_of_blocks = sim_dict["length_blocks"]

    """analysis of estimator properties and how the true beta got estimated"""
    beta_hat[beta_hat <= 0.1*amplitude] = 0
    number_correct_zero = sum((beta_hat == 0) & (true_beta== 0))

    beta_hat[(beta_hat >= 0.75*amplitude)  & (beta_hat <= 1.25*amplitude)] = amplitude
    number_correct = sum(beta_hat == amplitude)

    #count how many blocks got estimated correctly
    count = 0
    for j in range(num_simulations):
        for i in range(len(true_beta[:,1])-num_of_blocks):
            if sum(beta_hat[i:i+length_of_blocks,j] == amplitude) == length_of_blocks:
                count = count +1




    container_ana = [number_correct_zero, number_correct, count]
    with open(ppj("OUT_ANALYSIS", "analysis_fusion{}.pickle".format(sim_name)), "wb") as out_file:
       pickle.dump(container_ana, out_file)
