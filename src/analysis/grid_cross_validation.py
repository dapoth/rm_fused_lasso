"""
Perform cross-validation over a two-dimensional grid of penalty constants.

Cross-validation is performed with the sklearn package model_selection.GridSearchCV.

"""
import sys
import json
import pickle
from time import time
import numpy as np
from sklearn.model_selection import GridSearchCV
from bld.project_paths import project_paths_join as ppj
from src.model_code.flestimator import FusedLassoEstimator as fle

if __name__ == "__main__":
    
    SIM_NAME = sys.argv[2]
    REG_NAME = sys.argv[1]

    SIM_DICT = json.load(open(ppj("IN_MODEL_SPECS", SIM_NAME + ".json"),
                              encoding="utf-8"))

    with open(ppj("OUT_DATA", "data_{}.pickle".
                  format(SIM_NAME)), "rb") as in_file:
        BETA_X_EPSILON_Y = pickle.load(in_file)


    # Load data from json file.
    N_FEATURES = SIM_DICT["p"]
    N_OBS = SIM_DICT["n"]
    S1_MIN = SIM_DICT["s1_min"]
    S1_MAX = SIM_DICT["s1_max"]
    S2_MIN = SIM_DICT["s2_min"]
    S2_MAX = SIM_DICT["s2_max"]
    GRID_DENSITY = SIM_DICT['grid_density']

    # Load data from pickle file.
    X = BETA_X_EPSILON_Y[1]
    y = BETA_X_EPSILON_Y[3]


    # Construct two-dimensional fused-lasso grid for cross-validation.
    LASSO_GRID = {
        's1': list(np.linspace(S1_MIN, S1_MAX, GRID_DENSITY))
    }
    FUSION_GRID = {
        's2': list(np.linspace(S2_MIN, S2_MAX, GRID_DENSITY))
    }
    TWO_D_GRID = [{
        's1': list(np.linspace(S1_MIN, S1_MAX, GRID_DENSITY)),
        's2': list(np.linspace(S2_MIN, S2_MAX, GRID_DENSITY))
    }]

    # Construct two-dimensional lasso grid for cross-validation.
    if REG_NAME == 'lasso':
        LASSO_GRID = {
            's1': list(np.linspace(S1_MIN, SIM_DICT["s1_max_lasso"], GRID_DENSITY))
        }
        FUSION_GRID = {
            's2': list(np.linspace(11900, 12000, 1))
        }

        TWO_D_GRID = [{
            's1': list(np.linspace(S1_MIN, SIM_DICT["s1_max_lasso"], GRID_DENSITY)),
            's2': list(np.linspace(11900, 12000, 1))
        }]

    # Construct two-dimensional fusion grid for cross-validation.
    if REG_NAME == 'fusion':
        LASSO_GRID = {
            's1': list(np.linspace(11900, 12000, 1))
        }
        FUSION_GRID = {
            's2': list(np.linspace(S2_MIN, SIM_DICT["s2_max_fusion"], GRID_DENSITY))
        }
        TWO_D_GRID = [{
            's1': list(np.linspace(11900, 12000, 1)),
            's2': list(np.linspace(S2_MIN, SIM_DICT["s2_max_fusion"], GRID_DENSITY))
        }]


    START_TIME_CV = time()

    # Assess optimal penalty costants.
    CLF = GridSearchCV(fle(LASSO_GRID, FUSION_GRID), TWO_D_GRID,
                       scoring='neg_mean_squared_error',
                       n_jobs=-1, iid=False, refit=True,
                       cv=5, verbose=0, pre_dispatch='2*n_jobs',
                       error_score='raise-deprecating',
                       return_train_score='warn')

    CLF.fit(X, y[:, 1])
    PENALTY_CV = [CLF.best_params_["s1"], CLF.best_params_["s2"]]
    END_TIME_CV = time()

    # Save timings for fused lasso cross-validation.
    if REG_NAME == 'fused':

        TIMINGS_FUSED_LASSO = [N_FEATURES, N_OBS, np.round((END_TIME_CV - START_TIME_CV), 2)]

        with open(ppj("OUT_ANALYSIS", "time_list_{}.pickle".
                      format(SIM_NAME)), "wb") as out_file:
            pickle.dump(TIMINGS_FUSED_LASSO, out_file)


    CONTAINER = [CLF.cv_results_['mean_test_score'], CLF.cv_results_['params'], PENALTY_CV]
    with open(ppj("OUT_ANALYSIS", "cv_{}_{}.pickle".format(REG_NAME, SIM_NAME)), "wb") as out_file:
        pickle.dump(CONTAINER, out_file)
