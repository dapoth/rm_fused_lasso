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


if __name__ == "__main__":

    """ waf """
    sim_name = sys.argv[1]
    sim_dict = json.load(open(ppj("IN_MODEL_SPECS", sim_name + ".json"), encoding="utf-8"))
    with open(ppj("OUT_ANALYSIS", "simulation_{}.pickle".format(sim_name)), "rb") as in12_file:
        data = pickle.load(in12_file)



    """data import from pickle files"""
    beta = data[0]          #beta is p x num_simulations
    beta_hat = data[1]      #beta is p x num_simulations
    X = data[2]             # n x p
    X_T = data[3]           # p x n
    epsilion = data[4]      # n x num_simulations
    y = data[5]             # n x num_simulations
    num_simulations = sim_dict['num_simulations'] # how many times simulation gets run

    p = len(X[1,:])
    n = len(X[:,1])

    """building containers to store simulation results """

    beta_hat = np.empty((p,num_simulations))
    y_hat = np.empty((n,num_simulations))
    residuals = np.empty((n,num_simulations))


    """Calculation of optimal lambda (still missing)"""

    #s_1 = sim_dict['s1']
    #s_2 = sim_dict['s2']

    lasso_grid = {
      's1': list(np.linspace(sim_dict['s1_min'],sim_dict['s1_max'],20))
    }
    fused_grid = {
      's2': list(np.linspace(sim_dict['s2_min'],sim_dict['s2_max'],20))
    }

    two_d_grid = [{
                's1': list(np.linspace(sim_dict['s1_min'],sim_dict['s1_max'],20)),
                's2': list(np.linspace(sim_dict['s2_min'],sim_dict['s2_max'],20))
                }]

    clf = GridSearchCV(fle(lasso_grid,fused_grid), two_d_grid,
                            scoring= 'neg_mean_squared_error',
                            n_jobs = -1, iid= False, refit=True,
                            cv=None, verbose=0, pre_dispatch='2*n_jobs',
                            error_score='raise-deprecating',
                            return_train_score='warn')

    clf.fit(X, y[:,1])
    penalty_cv = [clf.best_params_ ["s1"], clf.best_params_["s2"]]
    opt_values = clf.best_params_ # dict of optimal parameters ["s1" : opt_value, "s2": opt_value]

    #np.savetxt("/home/christopher/Dokumente/Testbereich/s1.txt", np.array( [ opt_values["s1"], opt_values["s2"] ] )   )
    #np.savetxt("/home/christopher/Dokumente/Testbereich/beta_hat.txt", beta_hat)


    """calculation of beta to corresponding optimal lambda"""
    #for sim in range(num_simulations):
    #    beta_hat[:, sim] = fle(opt_values["s1"],opt_values["s2"]).fit( X,y[:, sim])

    for i in range(num_simulations):
        beta_hat[:,i] = fle(penalty_cv[0], penalty_cv[1]).fit(X, y[:,i])
        y_hat[:,i] = np.matmul(X,beta_hat[:,i])
        residuals[:,i] = y[:,i] - y_hat[:,i]

    container = [beta_hat, beta, penalty_cv, y_hat, residuals]
    with open(ppj("OUT_ANALYSIS", "beta_hat_{}.pickle".format(sim_name)), "wb") as out_file:
        pickle.dump(container, out_file)
        


    """analysis of estimator properties and how the true beta got estimated"""


    #container_ana = [placeholder, placeholde, placehold]
    #with open(ppj("OUT_ANALYSIS", "beta_hat_{}.pickle".format(sim_name)), "wb") as out_file:
    #    pickle.dump(container_ana, out_file)
