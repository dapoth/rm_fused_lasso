"""
For each of the four settings and for the lasso, fusion estimator and fused
lasso compute the mean squared error, the standard errors and the proportion of
correctly estimated blocks, nonzeros and zeros.

"""

import pickle
import json
import sys
import numpy as np
from bld.project_paths import project_paths_join as ppj


if __name__ == "__main__":
    
    SIM_NAME = sys.argv[2]
    REG_NAME = sys.argv[1]
    SIM_DICT = json.load(open(ppj("IN_MODEL_SPECS", SIM_NAME + ".json"),
                              encoding="utf-8"))
    with open(ppj("OUT_ANALYSIS", "simulation_{}_{}.pickle".
                  format(REG_NAME, SIM_NAME)), "rb") as in_file:
        SIMULATED_DATA = pickle.load(in_file)

    # Load data from pickle file.
    BETA_HAT = SIMULATED_DATA[0]
    TRUE_BETA = SIMULATED_DATA[1]
    RESIDUALS = SIMULATED_DATA[4]

    # Load data from json file.
    N_FEATURES = SIM_DICT["p"]
    N_OBS = SIM_DICT["n"]
    N_BLOCKS = SIM_DICT['number_of_blocks']
    LENGTH_BLOCKS = SIM_DICT['length_blocks']
    HEIGHT = SIM_DICT['amplitude']
    LEVELS = SIM_DICT['levels']
    SPIKES = SIM_DICT['spikes']
    N_SIMULATIONS = SIM_DICT['num_simulations']

    SIMULATION_RESULTS = []

    MSE = 1/N_OBS*np.sum(np.square(RESIDUALS), axis=0)

    MEAN_MSE = np.mean(MSE)
    STD_MSE = np.std(MSE)
    SIMULATION_RESULTS.append(MEAN_MSE)
    SIMULATION_RESULTS.append(STD_MSE)

    # Calculate percentage of relevant variables recognized.
    CORRECT_NONZERO = (sum((BETA_HAT >= 0.30*HEIGHT) & (TRUE_BETA > 0))
                       / np.sum(TRUE_BETA > 0, axis=0))
    SIMULATION_RESULTS.append(np.mean(CORRECT_NONZERO))
    SIMULATION_RESULTS.append(np.std(CORRECT_NONZERO))

    # Calculate percentage of correctly estimated zero coefficients.
    CORRECT_ZERO = (np.sum((np.absolute(BETA_HAT) <= 0.01) & (TRUE_BETA == 0), axis=0)
                    / np.sum(TRUE_BETA == 0, axis=0))
    SIMULATION_RESULTS.append(np.mean(CORRECT_ZERO))
    SIMULATION_RESULTS.append(np.std(CORRECT_ZERO))

    # Calculate percentage of correctly estimated blocks.
    # Avoid dividing by zero in the spike setting, in that case number_of_blocks is 0.
    if SIM_NAME == 'spikes':
        SIMULATION_RESULTS.append(np.NAN)
        SIMULATION_RESULTS.append(np.NAN)

    else:
        COUNTER_BLOCKS = np.sum(((BETA_HAT >= 0.50*HEIGHT) & (BETA_HAT <= 1.5*HEIGHT)
                                 & (TRUE_BETA == HEIGHT)) |
                                ((BETA_HAT >= 0.75*LEVELS) & (BETA_HAT <= 1.25*LEVELS)
                                 & (TRUE_BETA == LEVELS)), axis=0)

        PERCENT_BLOCKS = np.array(COUNTER_BLOCKS / (LENGTH_BLOCKS * N_BLOCKS))
        SIMULATION_RESULTS.append(np.mean(PERCENT_BLOCKS))
        SIMULATION_RESULTS.append(np.std(PERCENT_BLOCKS))



    with open(ppj("OUT_ANALYSIS", "analysis_{}_{}.pickle"
                  .format(REG_NAME, SIM_NAME)), "wb") as out_file:
        pickle.dump(SIMULATION_RESULTS, out_file)
