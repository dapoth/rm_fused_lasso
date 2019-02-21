# import sys
# import json
# import logging
# import pickle
# import numpy as np
# import cvxpy as cp
# from src.model_code.flestimator import FusedLassoEstimator as fle
# from src.model_code.functions import fused_lasso_primal
# from src.model_code.functions import generate_blocks
# from bld.project_paths import project_paths_join as ppj
# from sklearn.model_selection import GridSearchCV
# import random
#
# if __name__ == "__main__":
#
#     """ waf """
#     mont_name = sys.argv[0]
#     sim_dict = json.load(open(ppj("IN_MODEL_SPECS", "one_block_monte_carlo.json"), encoding="utf-8"))
#
#
#
#     beta_container = np.ones((sim_dict["p"], sim_dict["num_simulations"], len(sim_dict["n"])))
#     s_opt_container = np.zeros([2,len(sim_dict["n"])])
#
#
#
#     beta = generate_blocks(sim_dict["p"], sim_dict["number_of_blocks"], sim_dict["length_blocks"], sim_dict["amplitude"],
#                         sim_dict["spike_level"])
#
#
#     for k in list(range(len(sim_dict["n"]))):
#
#         mean_x = np.zeros(sim_dict["p"])
#         cov_X = np.identity(sim_dict["p"])
#         X = np.random.multivariate_normal(mean_x, cov_X,sim_dict["n"][k])
#
#         eps = np.random.randn(sim_dict["n"][k])
#
#         y = X.dot(beta)+eps
#
#         lasso_grid = {
#           's1': list(np.linspace(sim_dict['s1_min'],sim_dict['s1_max'],20))
#         }
#         fused_grid = {
#           's2': list(np.linspace(sim_dict['s2_min'],sim_dict['s2_max'],20))
#         }
#
#         two_d_grid = [{
#                     's1': list(np.linspace(sim_dict['s1_min'],sim_dict['s1_max'],20)),
#                     's2': list(np.linspace(sim_dict['s2_min'],sim_dict['s2_max'],20))
#                     }]
#
#         clf = GridSearchCV(fle(lasso_grid,fused_grid), two_d_grid,
#                                 scoring= 'neg_mean_squared_error',
#                                 n_jobs = -1, iid= False, refit=True,
#                                 cv=None, verbose=0, pre_dispatch='2*n_jobs',
#                                 error_score='raise-deprecating',
#                                 return_train_score='warn')
#
#         clf.fit(X, y)
#
#
#         s1 = clf.best_params_ ["s1"]
#         s2 = clf.best_params_["s2"]
#
#         s_opt_container[0,k] = s1
#         s_opt_container[1,k] = s2
#         for index ,i in enumerate(sim_dict["n"]):  # i =10 i = 1000 usw
# 
#             mean_x = np.zeros(sim_dict["p"])
#             cov_X = np.identity(sim_dict["p"])
#             X = np.random.multivariate_normal(mean_x, cov_X,sim_dict["n"][k])
#
#             for j in range(sim_dict["num_simulations"]):
#
#                 eps = np.random.rand(sim_dict["n"][k])
#
#                 y = np.matmul(X,beta)+eps
#
#                 beta_container[:,j,index] = fused_lasso_primal(y,X,s1,s2)
#
#     test1234 = [beta, beta_container, s_opt_container]
#
#     with open(ppj("OUT_ANALYSIS", "beta_hat_monte_Carlo_{}.pickle".format(mont_name)), "wb") as out_file:
#         pickle.dump(test1234, out_file)
