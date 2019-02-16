"""Draw simulated samples from two uncorrelated uniform variables
(locations in two dimensions) for two types of agents and store
them in a 3-dimensional NumPy array.

*Note:* In principle, one would read the number of dimensions etc.
from the "IN_MODEL_SPECS" file, this is to demonstrate the most basic
use of *run_py_script* only.

"""

import numpy as np
from bld.project_paths import project_paths_join as ppj
import json
import pickle
import math
import random as rd


np.random.seed(12345)

def generate_blocks(p, number_blocks, length_blocks, amplitude,  spike_level, levels = False, spikes = 0):
    """
    generate beta's for simulation purpose.
    """

    container = np.zeros(p)
    max_blocks = math.floor(p/ length_blocks)

    #blocks = np.linspace(1, number_blocks, number_blocks)
    start_blocks = rd.sample(range(max_blocks),number_blocks)

#    if max_blocks < number_blocks:
#        break

    amplitudes = [amplitude, amplitude*2]

    if levels == True :
        """
        If the Blocks should not all have equal levels, we will randomly chose
        the level of each block as either amplitude or amplitude times 2.
        """

        for block in start_blocks :
            amp = rd.choice(amplitudes)
            for i in range(p) :
                if ( i > ( block-1)* length_blocks) and ( i <= block* length_blocks):
                    container [i] = amp

    else:
        for block in start_blocks :
            #amp = rd.choice(amplitudes)
            for i in range(p) :
                if ( i > ( block-1)* length_blocks) and ( i <= block* length_blocks):
                    container [i] = amplitude

    #if spikes != 0 :
     #   rd.choice(container[:]==0, spikes) = spike_level

    return container


def generate_data():
    beta = np.zeros(p)
    eps = np.random.randn(n)
    mean = np.ones(p)
    cov = np.identity(p)
    x = np.random.multivariate_normal(mean, cov,n)
    y = np.matmul(x,beta)+eps
    return y,x,beta,eps


def save_y(y):
    y.tofile(ppj("OUT_DATA", "y.csv"), sep=",")
    #np.savetxt(ppj("OUT_DATA", "y.csv"),y)
def save_x(x):
    x.tofile(ppj("OUT_DATA", "x.csv"), sep=",")
def save_beta(beta):
    beta.tofile(ppj("OUT_DATA", "beta.csv"), sep=",")
def save_eps(eps):
    eps.tofile(ppj("OUT_DATA", "eps.csv"), sep=",")

def generate_data1(n,p,number_blocks, length_blocks, amplitude,  spike_level, levels = False, spikes = 0):

    beta = np.zeros([p,num_simulations])
    for sim in range(num_simulations):
        beta[:,sim] = generate_blocks(p, number_blocks, length_blocks, amplitude,  spike_level, levels = False, spikes = 0)
    beta_hat = np.ones([p,num_simulations])
    mean = np.zeros(p)
    cov = np.identity(p)
    X = np.random.multivariate_normal(mean, cov,n)
    X_t = np.transpose(X)
    mean_eps = np.zeros(num_simulations)
    cov_eps = np.identity(num_simulations)
    epsilon = np.random.multivariate_normal(mean_eps, cov_eps,n)
    Y = np.matmul(X, beta) + epsilon

    return beta, beta_hat,X,X_t,epsilon,Y





if __name__ == "__main__":
    data_simulation = json.load(open(ppj("IN_MODEL_SPECS", "data_simulation.json"), encoding="utf-8"))
    n = data_simulation['n']
    p = data_simulation['p']
    y,x,beta,eps = generate_data()
    save_y(y)
    save_x(x)
    save_beta(beta)
    save_eps(eps)

    