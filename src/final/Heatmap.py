import numpy as np
import matplotlib.pyplot as plt
from bld.project_paths import project_paths_join as ppj
import sys
import pickle
import seaborn
import json

if __name__ == "__main__":
    """Heatmap."""
    sim_name = sys.argv[1]

    sim_dict = json.load(open(ppj("IN_MODEL_SPECS", sim_name + ".json"),
                         encoding="utf-8"))
    with open(ppj("OUT_ANALYSIS", "simulation_fused_{}.pickle".
                  format(sim_name)), "rb") as in12_file:
        simulated_data = pickle.load(in12_file)

    mean_test_score = simulated_data[5]
    parameters = simulated_data[6]
    grid_density = sim_dict['grid_density']

    # Array in die richtige 2-dimensionale Form bringen.
    new_array = np.reshape(np.abs(mean_test_score), (grid_density, grid_density))
    new_array_sorted = np.flip(new_array)

    # Extract s1 and s2 value for axis labeling
    lists2 = np.zeros(grid_density)
    for i in range(grid_density):
        lists2[i] = round(parameters[i]['s2'], 1)

    lists1 = np.zeros(len(mean_test_score))
    for i in range(len(mean_test_score)):
        lists1[i] = round(parameters[i]['s1'], 1)
    lists1unique = np.flip(np.unique(lists1))

    # Create Heatmap
    heatmap = seaborn.heatmap(new_array_sorted,
                              xticklabels=lists2, yticklabels=lists1unique)
    plt.savefig(ppj("OUT_FIGURES", "heatmap_{}.png".format(sim_name)))

    # Changing colour of the heatmap
    cmap = "YlGnBu"
    cmap = "RdYlGn"
    cmap = "RdYlGn"
    linewidths = .1
