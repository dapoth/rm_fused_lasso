#import math
import random as rd
import numpy as np
from numba import jit
import math

@jit(nopython=True)
def generate_blocks(num_features, number_blocks, length_blocks, block_height,
                    spike_height=7, levels=False, spikes=0):
    """Generate non-overlapping *generate_blocks* for one simulation step.

    Divide the generated_blocks into 1/*length_blocks* possible blocks and
    randomly setting *number_blocks* of them non-zero.

    Args:
        num_features (int): number of features
        number_blocks (int): non overlapping feature blocks to create
        length_blocks (int): the length of feature blocks to create
        block_height (int): height of basis feature blocks
        spike_height (int): height of spikes
        levels (boolean): indicate whether blocks should two different
            possible heights
        spikes (int): number of spikes to be added

    Returns:
        generated_blocks (np.ndarray)

    """
    max_blocks = np.int(np.floor(num_features / length_blocks))
    if max_blocks < number_blocks:
        raise TypeError("""The number of blocks must not exceed the maximal number 
                        of blocks possible for given p and length of blocks.""")
    generated_blocks = np.zeros(num_features)
    start_blocks = np.random.choice(np.arange(max_blocks), number_blocks, replace=False)
    
    if levels:
        """
        If the Blocks should not all have equal levels, we will randomly chose
        the level of each block as either block_height or block_height *2.
        """
        heights = np.array([block_height, block_height*2])
        for block in start_blocks:
            random_height = np.random.choice(heights)
            for i in range(num_features):
                if (i >= block * length_blocks) and (i < (block+1) *
                                                     length_blocks):
                    generated_blocks[i] = random_height

    else:
        for block in start_blocks:
            for i in range(num_features):
                if (i >= block * length_blocks) and (i < (block+1) *
                                                     length_blocks):
                    generated_blocks[i] = block_height

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
