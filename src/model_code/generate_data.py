import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pandas as pd
import random as rd
from sklearn import model_selection
import math
from src.model_code.generate_blocks import generate_blocks


def generate_data(n, p, num_simulations, number_blocks, length_blocks,
                  amplitude, spike_level, levels=False, spikes=0):
    """
    Together with the generate_blocks function, it generates the data for
    our simulations.
    """

    beta = np.zeros([p, num_simulations])
    for sim in range(num_simulations):
        beta[:, sim] = generate_blocks(p, number_blocks, length_blocks,
                                       amplitude, spike_level, levels, spikes)
    mean = np.zeros(p)
    cov = np.identity(p)
    X = np.random.multivariate_normal(mean, cov, n)
    mean_eps = np.zeros(num_simulations)
    cov_eps = np.identity(num_simulations)
    epsilon = np.random.multivariate_normal(mean_eps, cov_eps, n)
    Y = np.matmul(X, beta) + epsilon

    return beta, X, epsilon, Y
