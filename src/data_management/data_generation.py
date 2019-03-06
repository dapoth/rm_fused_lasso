"""
Generate data for all four settings with the help of the generate_data function
and the model parameters for each seetting as specified in the JSON files.

"""

import pickle
import sys
import json
import numpy as np
from bld.project_paths import project_paths_join as ppj
#from src.model_code.generate_data import generate_data
from generate_data import generate_data

np.random.seed(12345)
import ast



if __name__ == "__main__":
    #reg_name = sys.argv[1]
    sim_name = sys.argv[1]
    sim = json.load(open(ppj("IN_MODEL_SPECS", sim_name + ".json"),
                    encoding="utf-8"))



    ##### Load Model specs
    num_simulations = sim['num_simulations']
    p = sim['p']
    n = sim['n']
    number_blocks = sim['number_of_blocks']
    length_blocks = sim['length_blocks']
    amplitude = sim['amplitude']
    spike_level = sim['spike_level']
    levels = sim['levels']
    spikes = sim['spikes']

    [beta, X, epsilon, Y] = generate_data(num_simulations,n,p, number_blocks,
                                                          length_blocks, amplitude,  spike_level, levels, spikes)

    aux1 = [beta, X, epsilon, Y]

    with open(ppj("OUT_DATA", "data_{}.pickle".format(sim_name)), "wb") as out12_file:
        pickle.dump(aux1, out12_file)
