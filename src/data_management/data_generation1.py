"""Docstring."""

import pickle
import math
import random as rd
import sys
import json
import numpy as np
from bld.project_paths import project_paths_join as ppj
from src.model_code.functions import generate_data1
from src.model_code.functions import generate_blocks
np.random.seed(12345)


if __name__ == "__main__":
    sim_name = sys.argv[1]
    sim = json.load(open(ppj("IN_MODEL_SPECS", sim_name + ".json"),
                    encoding="utf-8"))

    num_simulations = sim['num_simulations']
    p = sim['p']
    n = sim['n'][1]
    number_blocks = sim['number_of_blocks']
    length_blocks = sim['length_blocks']
    amplitude = sim['amplitude']
    spike_level = sim['spike_level']
    levels = sim['levels']

    #this function call returns the following elements: beta, beta_hat, X, X_t, epsilon, Y
    [beta, beta_hat, X, X_t, epsilon, Y] = generate_data1(n, p, num_simulations, number_blocks,
                                                          length_blocks, amplitude,  spike_level, levels, spikes=0 )

    aux1 = [beta, beta_hat, X, X_t, epsilon, Y]

    with open(ppj("OUT_ANALYSIS", "simulation_{}.pickle".format(sim_name)), "wb") as out12_file:
        pickle.dump(aux1, out12_file)
