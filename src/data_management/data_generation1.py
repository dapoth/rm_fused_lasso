"""Docstring."""

import pickle
import math
import random as rd
import sys
import json
import numpy as np
from bld.project_paths import project_paths_join as ppj


np.random.seed(12345)


def generate_blocks(p, number_blocks, length_blocks, amplitude,
                    spike_level, levels=False, spikes=0):
    """Generate beta's for simulation purpose."""
    container = np.zeros(p)
    max_blocks = math.floor(p / length_blocks)

    # blocks = np.linspace(1, number_blocks, number_blocks)
    start_blocks = rd.sample(range(max_blocks), number_blocks)

#    if max_blocks < number_blocks:
#        break

    amplitudes = [amplitude, amplitude*2]

    if (levels == True):
        """
        If the Blocks should not all have equal levels, we will randomly chose
        the level of each block as either amplitude or amplitude times 2.
        """

        for block in start_blocks :
            amp = rd.choice(amplitudes)
            for i in range(p) :
                if (i > (block-1) * length_blocks) and (i <= block * length_blocks):
                    container[i] = amp

    else:
        for block in start_blocks :
            #amp = rd.choice(amplitudes)
            for i in range(p) :
                if ( i > ( block-1)* length_blocks) and (i <= block* length_blocks):
                    container [i] = amplitude

    #if spikes != 0 :
     #   rd.choice(container[:]==0, spikes) = spike_level

    return container


def generate_data1(n, p, number_blocks, length_blocks, amplitude, spike_level,
                   levels=False, spikes=0):
    """Docstr"""
    beta = np.zeros([p, num_simulations])
    for sim in range(num_simulations):
        beta[:, sim] = generate_blocks(p, number_blocks, length_blocks, amplitude, spike_level, levels=False, spikes=0)
    beta_hat = np.ones([p, num_simulations])
    mean = np.zeros(p)
    cov = np.identity(p)
    X = np.random.multivariate_normal(mean, cov, n)
    X_t = np.transpose(X)
    mean_eps = np.zeros(num_simulations)
    cov_eps = np.identity(num_simulations)
    epsilon = np.random.multivariate_normal(mean_eps, cov_eps, n)
    Y = np.matmul(X, beta) + epsilon

    return beta, beta_hat, X, X_t, epsilon, Y


if __name__ == "__main__":
    sim_name = sys.argv[1]
    sim = json.load(open(ppj("IN_MODEL_SPECS", sim_name + ".json"),
                    encoding="utf-8"))

    num_simulations = sim['num_simulations']
    p = sim['p']
    n = sim['n']
    number_blocks = sim['number_of_blocks']
    length_blocks = sim['length_blocks']
    amplitude = sim['amplitude']
    spike_level = sim['spike_level']
    levels = sim['levels']

    [beta, beta_hat, X, X_t, epsilon, Y] = generate_data1(n, p, number_blocks,
                                                          length_blocks, amplitude,  spike_level, levels, spikes=0)

    aux1 = [beta, beta_hat, X, X_t, epsilon, Y]

    with open(ppj("OUT_ANALYSIS", "simulation_{}.pickle".format(sim_name)), "wb") as out12_file:
        pickle.dump(aux1, out12_file)
