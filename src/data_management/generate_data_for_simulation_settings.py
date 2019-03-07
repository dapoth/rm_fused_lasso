"""Generate data for all four settings.

Generate data with the generate_data function and model parameters for each
setting as specified in the json files.

"""
import pickle
import sys
import json
import numpy as np
from bld.project_paths import project_paths_join as ppj
from functions_for_data_generation import generate_data


if __name__ == "__main__":

    SIN_NAME = sys.argv[1]
    SIM = json.load(open(ppj("IN_MODEL_SPECS", SIN_NAME + ".json"),
                         encoding="utf-8"))
    np.random.seed(420)

    # Load data from json file.
    N_SIMULATIONS = SIM['num_simulations']
    N_FEATURES = SIM['p']
    N_OBS = SIM['n']
    N_BLOCKS = SIM['number_of_blocks']
    LENGTH_BLOCKS = SIM['length_blocks']
    HEIGHT_BLOCKS = SIM['amplitude']
    HEIGHT_SPIKE = SIM['spike_level']
    LEVELS = SIM['levels']
    SPIKES = SIM['spikes']

    # Generate data with loaded information.
    [BETA, X, EPSILON, Y] = generate_data(N_SIMULATIONS, N_OBS, N_FEATURES,
                                          N_BLOCKS, LENGTH_BLOCKS, HEIGHT_BLOCKS,
                                          HEIGHT_SPIKE, LEVELS, SPIKES)
    DATASET = [BETA, X, EPSILON, Y]

    with open(ppj("OUT_DATA", "data_{}.pickle".format(SIN_NAME)), "wb") as out_file:
        pickle.dump(DATASET, out_file)
