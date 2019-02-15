import json
import pickle
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from bld.project_paths import project_paths_join as ppj

PLOT_ARGS = {"markersize": 4, "alpha": 0.6}


def plot_locations(aux, model_name):
    
    #fig = plt.subplot(np.linspace(0, 10, 10),aux)
    fig = plt.figure()
    plt.plot(range(1,11),aux)
    fig.savefig(ppj("OUT_FIGURES", "schelling_{}.png".format(model_name)))


if __name__ == "__main__":
    model_name = sys.argv[1]
    model = json.load(open(ppj("IN_MODEL_SPECS", model_name + ".json"), encoding="utf-8"))

    # Load locations after each round
    with open(ppj("OUT_ANALYSIS", "schelling_{}.pickle".format(model_name)), "rb") as in_file:
        aux = pickle.load(in_file)

    plot_locations(aux, model_name)
