"""Plot heatmaps of gridCV for the calculated mean squared errors and grid (s1,s2) values."""
import sys
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from bld.project_paths import project_paths_join as ppj


if __name__ == "__main__":

    SIM_NAME = sys.argv[1]

    SIM_DICT = json.load(open(ppj("IN_MODEL_SPECS", SIM_NAME + ".json"),
                              encoding="utf-8"))
    with open(ppj("OUT_ANALYSIS", "simulation_fused_{}.pickle".
                  format(SIM_NAME)), "rb") as in_file:
        SIM_DATA = pickle.load(in_file)

    MEAN_TEST_SCORE = SIM_DATA[5]
    PARAMETERS = SIM_DATA[6]
    GRID_DENSITY = SIM_DICT['grid_density']

    # Adapt 1d-array of test scores to a 2d-array for the heatmap.
    TEST_SCORE_MATRIX = np.reshape(np.abs(MEAN_TEST_SCORE), (GRID_DENSITY, GRID_DENSITY))
    TEST_SCORE_MATRIX = np.flip(TEST_SCORE_MATRIX)

    # Extract s1 and s2 value for axis labeling
    S2_GRID = np.zeros(GRID_DENSITY)
    for i in range(GRID_DENSITY):
        S2_GRID[i] = round(PARAMETERS[i]['s2'], 1)

    S1_GRID = np.zeros(len(MEAN_TEST_SCORE))
    for i in range(len(MEAN_TEST_SCORE)):
        S1_GRID[i] = round(PARAMETERS[i]['s1'], 1)
    S1_GRID_UNI = np.flip(np.unique(S1_GRID))   #Values repeat them self, therefore unique.

    # Create heatmap
    HEATMAP = seaborn.heatmap(TEST_SCORE_MATRIX,
                              xticklabels=S2_GRID, yticklabels=S1_GRID_UNI)
    plt.savefig(ppj("OUT_FIGURES", "heatmap_{}.png".format(SIM_NAME)))
