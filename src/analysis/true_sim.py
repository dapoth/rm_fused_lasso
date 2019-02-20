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

    specifity_zero = beta_hat[beta_hat <= 0.1*sim_dict["amplitude"]] = 1
    number_correct_zero = sum(specifity_zero, start=None)
    specifity_block = beta_hat[ beta_hat <= 0,75*sim_dict["amplitude"]  and beta_hat <= 1,25*sim_dict["amplitude"]] = 1
    number_correct = sum(specifity_block)
    comparison = beta == beta_hat
    sum_comparison = sum(comparison[:,1])
    print(sum_comparison)


    #container_ana = [placeholder, placeholde, placehold]
    #with open(ppj("OUT_ANALYSIS", "beta_hat_{}.pickle".format(sim_name)), "wb") as out_file:
    #    pickle.dump(container_ana, out_file)

    beta_container = np.ones((sim_dict["p"], sim_dict["num_simulations_mont"], len(sim_dict["n"])))
    s_opt_container = np.zeros([2,len(sim_dict["n"])])



    beta = generate_blocks(sim_dict["p"], sim_dict["number_of_blocks"], sim_dict["length_blocks"], sim_dict["amplitude"],
                        sim_dict["spike_level"])


    for k in list(range(len(sim_dict["n"]))):

        mean_x = np.zeros(sim_dict["p"])
        cov_X = np.identity(sim_dict["p"])
        X = np.random.multivariate_normal(mean_x, cov_X,sim_dict["n"][k])

        eps = np.random.randn(sim_dict["n"][k])

        y = X.dot(beta)+eps

        lasso_grid = {
          's1': list(np.linspace(sim_dict['s1_min'],sim_dict['s1_max'],20))
        }
        fused_grid = {
          's2': list(np.linspace(sim_dict['s2_min'],sim_dict['s2_max'],20))
        }

        two_d_grid = [{
                    's1': list(np.linspace(sim_dict['s1_min'],sim_dict['s1_max'],10)),
                    's2': list(np.linspace(sim_dict['s2_min'],sim_dict['s2_max'],10))
                    }]

        clf = GridSearchCV(fle(lasso_grid,fused_grid), two_d_grid,
                                scoring= 'neg_mean_squared_error',
                                n_jobs = -1, iid= False, refit=True,
                                cv=None, verbose=0, pre_dispatch='2*n_jobs',
                                error_score='raise-deprecating',
                                return_train_score='warn')

        clf.fit(X, y)


        s1 = clf.best_params_ ["s1"]
        s2 = clf.best_params_["s2"]

        s_opt_container[0,k] = s1
        s_opt_container[1,k] = s2
        for index ,i in enumerate(sim_dict["n"]):  # i =10 i = 1000 usw

            mean_x = np.zeros(sim_dict["p"])
            cov_X = np.identity(sim_dict["p"])
            X = np.random.multivariate_normal(mean_x, cov_X,sim_dict["n"][k])

            for j in range(sim_dict["num_simulations_mont"]):

                eps = np.random.rand(sim_dict["n"][k])

                y = np.matmul(X,beta)+eps

                beta_container[:,j,index] = fused_lasso_primal(y,X,s1,s2)

    test1234 = [beta, beta_container, s_opt_container]

    with open(ppj("OUT_ANALYSIS", "beta_hat_monte_Carlo_{}.pickle".format(sim_name)), "wb") as out_file:
        pickle.dump(test1234, out_file)
