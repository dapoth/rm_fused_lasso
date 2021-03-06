import math
import random as rd
import numpy as np
from numba import jit


@jit(nopython=True)
def generate_beta(num_features, number_blocks, length_blocks, block_height,
                  levels=False, spikes=0, spike_height=7):
    """Generate beta with non-overlapping blocks.

    Divide the generated_blocks into 1/*length_blocks* possible blocks and
    randomly setting *number_blocks* of them non-zero.

    Args:
        | num_features (int): number of features
        | number_blocks (int): non overlapping feature blocks to create
        | length_blocks (int): the length of feature blocks to create
        | block_height (int): height of basis feature blocks
        | spike_height (int): height of spikes
        | levels (boolean): indicate whether blocks should have two
            different possible heights
        | spikes (int): number of spikes to be added

    Returns:
        | generated_blocks (np.ndarray)

    """
    max_blocks = math.floor(num_features / length_blocks)
    if max_blocks < number_blocks:
        raise ValueError("""The number of blocks must not exceed the maximal number
                        of blocks possible for given p and length of blocks.""")

    non_zero = number_blocks* length_blocks + spikes
    if num_features < non_zero:
        raise ValueError("""The number of nonzero coefficients must not exceed p.""")

    generated_blocks = np.zeros(num_features)
    start_blocks = np.random.choice(np.arange(max_blocks), number_blocks, replace=False)

    # If the blocks should not all have equal levels, randomly chose
    # block level from block_height and block_height *2.
    if levels:

        heights = np.array([block_height, block_height*2])
        for block in start_blocks:
            random_height = np.random.choice(heights)
            lower_bound = block * length_blocks
            upper_bound = (block+1) * length_blocks
            for i in range(lower_bound, upper_bound):
                generated_blocks[i] = random_height

    else:
        for block in start_blocks:
            lower_bound = block * length_blocks
            upper_bound = (block+1) * length_blocks
            for i in range(lower_bound, upper_bound):
                generated_blocks[i] = block_height

    # Set spikes if they should be inculded.
    if spikes != 0:
        non_blocks = []
        for i in range(num_features):
            if generated_blocks[i] == 0:
                non_blocks.append(i)
        non_blocks_array = np.array(non_blocks)
        beta_spikes = np.random.choice(non_blocks_array, spikes, replace=False)
        for k in range(len(beta_spikes)):
            beta_spikes[k] = int(beta_spikes[k])
        for i in beta_spikes:
            generated_blocks[int(i)] = spike_height

    return generated_blocks



def generate_data(num_simulations, num_observations, num_features, num_blocks,
                  length_blocks, height, spike_height, levels=False, spikes=0):
    """Generate block sparse data samples to perform fused lasso simulations.

    Args:
        | num_features (int): number of features in each simulation step
        | num_blocks (int): non overlapping feature blocks to create
        | length_blocks (int): the length of feature blocks to create
        | block_height (int): height of basis feature blocks
        | spike_height (int): height of spikes
        | levels (boolean): indicate whether blocks should have two
            different possible heights
        | spikes (int): number of spikes to be added

    Returns:
        | beta (np.ndarray)
        | X (np.ndarray)
        | epsilon (np.ndarray)
        | y (np.ndarray)

    """
    beta = np.zeros([num_features, num_simulations])
    for sim in range(num_simulations):
        beta[:, sim] = generate_beta(num_features, num_blocks, length_blocks,
                                     height, levels, spikes, spike_height)
    mean = np.zeros(num_features)
    cov = np.identity(num_features)
    X = np.random.multivariate_normal(mean, cov, num_observations)
    mean_eps = np.zeros(num_simulations)
    cov_eps = np.identity(num_simulations)
    epsilon = np.random.multivariate_normal(mean_eps, cov_eps, num_observations)
    y = np.matmul(X, beta) + epsilon

    return beta, X, epsilon, y
