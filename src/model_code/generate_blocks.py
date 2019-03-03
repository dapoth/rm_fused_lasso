import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pandas as pd
import random as rd
from sklearn import model_selection
import math

def generate_blocks( p, number_blocks, length_blocks, block_height,
                    spike_height, levels=False, spikes=0):
    """Generate non-overlapping *generate_blocks* for one simulation step.

    Divide the generated_blocks into 1/*length_blocks* possible blocks and
    randomly setting *number_blocks* of them non-zero.

    Args:
        p (int): number of features
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
    generated_blocks = np.zeros(p)
    max_blocks = math.floor(p / length_blocks)

    start_blocks = rd.sample(range(max_blocks), number_blocks)

#    if max_blocks < number_blocks:
#        break

    if levels:
        """
        If the Blocks should not all have equal levels, we will randomly chose
        the level of each block as either block_height or block_height *2.
        """
        heights = [block_height, block_height*2]
        for block in start_blocks:
            random_height = rd.choice(heights)
            for i in range(p):
                if (i >= block * length_blocks) and (i < (block+1) *
                                                     length_blocks):
                    generated_blocks[i] = random_height

    else:
        for block in start_blocks:
            for i in range(p):
                if (i >= block * length_blocks) and (i < (block+1) *
                                                     length_blocks):
                    generated_blocks[i] = block_height

    if spikes != 0:
        non_blocks = []
        for i in range(p):
            if generated_blocks[i] == 0:
                non_blocks.append(i)
        beta_spikes = rd.sample(non_blocks, spikes)
        for i in beta_spikes:
            generated_blocks[i] = spike_height

    return generated_blocks
