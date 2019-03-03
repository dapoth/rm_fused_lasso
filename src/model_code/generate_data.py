import numpy as np
from src.model_code.generate_blocks import generate_blocks


def generate_data(num_simulations, num_observations, num_features, num_blocks,
                  length_blocks, height, spike_height, levels=False, spikes=0):
    """Generate block sparse data samples to perform fused lasso simulations.

    Args:
        num_features (int): number of features in each simulation step
        num_blocks (int): non overlapping feature blocks to create
        length_blocks (int): the length of feature blocks to create
        block_height (int): height of basis feature blocks
        spike_height (int): height of spikes
        levels (boolean): indicate whether blocks should two different
            possible heights
        spikes (int): number of spikes to be added

    Returns:
        beta (np.ndarray)
        X (np.ndarray)
        epsilon (np.ndarray)
        y (np.ndarray)

    """
    beta = np.zeros([num_features, num_simulations])
    for sim in range(num_simulations):
        beta[:, sim] = generate_blocks(num_features, num_blocks, length_blocks,
                                       height, spike_height, levels, spikes)
    mean = np.zeros(num_features)
    cov = np.identity(num_features)
    X = np.random.multivariate_normal(mean, cov, num_observations)
    mean_eps = np.zeros(num_simulations)
    cov_eps = np.identity(num_simulations)
    epsilon = np.random.multivariate_normal(mean_eps, cov_eps, num_observations)
    y = np.matmul(X, beta) + epsilon

    return beta, X, epsilon, y
