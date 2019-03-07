"""
Perform estimation of betas for all simulation settings and different
estimation methods.

"""
import sys
import json
import pickle
from time import time
import numpy as np
from src.model_code.flestimator import FusedLassoEstimator as fle
from src.model_code.fused_lasso_primal import fused_lasso_primal
from bld.project_paths import project_paths_join as ppj

if __name__ == "__main__":
    
    REG_NAME = sys.argv[1]
    SIM_NAME = sys.argv[2]
    SIM_DICT = json.load(open(ppj("IN_MODEL_SPECS", SIM_NAME + ".json"),
                              encoding="utf-8"))
    with open(ppj("OUT_DATA", "data_{}.pickle".
                  format(SIM_NAME)), "rb") as in_file:
        BETA_X_EPSILON_Y = pickle.load(in_file)

    with open(ppj("OUT_ANALYSIS", "cv_{}_{}.pickle".
                  format(REG_NAME, SIM_NAME)), "rb") as in_file:
        CROSSVALIDATION_RESULTS = pickle.load(in_file)


    # Load data from pickle file.
    TRUE_BETA = BETA_X_EPSILON_Y[0]         # p x num_simulations
    X = BETA_X_EPSILON_Y[1]                 # n x p
    y = BETA_X_EPSILON_Y[3]                 # n x num_simulations

    # Load data from pickle file.
    MSE = CROSSVALIDATION_RESULTS[0]
    PARAMETER_GRID = CROSSVALIDATION_RESULTS[1]
    S1_OPT = CROSSVALIDATION_RESULTS[2][0]
    S2_OPT = CROSSVALIDATION_RESULTS[2][1]

    # Load data from json file.
    N_FEATURES = SIM_DICT["p"]
    N_OBS = SIM_DICT["n"]
    N_SIMULATIONS = SIM_DICT['num_simulations']

    # Building containers to store simulation results.
    BETA_HAT = np.empty((N_FEATURES, N_SIMULATIONS))
    Y_HAT = np.empty((N_OBS, N_SIMULATIONS))
    RESIDUALS = np.empty((N_OBS, N_SIMULATIONS))


    # Calculate beta estimates for cross-validated penalty terms.

    for i in range(N_SIMULATIONS):
        start_time_estimation = time()
        BETA_HAT[:, i] = fused_lasso_primal(y[:, i], X, S1_OPT, S2_OPT)
        #fle(penalty_cv[0], penalty_cv[1]).fit(X, y[:, i])
        Y_HAT[:, i] = np.matmul(X, BETA_HAT[:, i])
        RESIDUALS[:, i] = y[:, i] - Y_HAT[:, i]
        end_time_estimation = time()

    # Save timings for fused lasso setting.
    if REG_NAME == 'fused':

        with open(ppj("OUT_ANALYSIS", "time_list_{}.pickle".
                      format(SIM_NAME)), "rb") as in_file:
            TIMINGS = pickle.load(in_file)

        TIME_ESTIMATION = np.round((end_time_estimation - start_time_estimation), 2)
        TIMINGS.append(TIME_ESTIMATION)

        with open(ppj("OUT_ANALYSIS", "time_list_{}.pickle".
                      format(SIM_NAME)), "wb") as out_file:
            pickle.dump(TIMINGS, out_file)

    # Save estimation results.
    PENALTY_CV = [S1_OPT, S2_OPT]
    CONTAINER = [BETA_HAT, TRUE_BETA, PENALTY_CV, Y_HAT, RESIDUALS,
                 MSE, PARAMETER_GRID]
    with open(ppj("OUT_ANALYSIS", "simulation_{}_{}.pickle".
                  format(REG_NAME, SIM_NAME)), "wb") as out_file:
        pickle.dump(CONTAINER, out_file)
