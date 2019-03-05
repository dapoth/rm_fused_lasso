#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 23:49:37 2019

@author: clara
"""
import math
import random as rd
import numpy as np
from numba import jit
from time import time


num_features = 2000
number_blocks =  3
block_height= 3
length_blocks= 9
spikes = 3
spike_height =  7
levels =  1

@jit(nopython=True)
def generate_blocks(num_features, number_blocks, length_blocks, block_height,
                    spike_height, levels=False, spikes=0):
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

    generated_blocks = np.zeros(num_features)
    max_blocks = np.floor(num_features / length_blocks)


    start_blocks = np.random.choice(np.arange(max_blocks), number_blocks)


#    if max_blocks < number_blocks:
#        break

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
        non_blocks = np.arange(num_features-number_blocks * length_blocks - spikes)
        for i in range(num_features):
            if generated_blocks[i] == 0:
                for j in range(len(non_blocks)):
                    non_blocks[j] = i
        beta_spikes = np.random.choice(non_blocks, spikes)
        for k in range(len(beta_spikes)):
            beta_spikes[k] = int(beta_spikes[k])
        for i in beta_spikes:
            generated_blocks[i] = spike_height

    return generated_blocks

start_time_new = time()
generate_blocks_new(num_features, number_blocks, length_blocks, block_height, spike_height, levels, spikes)
end_time_new = time()

print( end_time_new-start_time_new)


def generate_blocks(num_features, number_blocks, length_blocks, block_height,
                    spike_height, levels=False, spikes=0):
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
    generated_blocks = np.zeros(num_features)
    max_blocks = math.floor(num_features / length_blocks)

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
        beta_spikes = rd.sample(non_blocks, spikes)
        for i in beta_spikes:
            generated_blocks[i] = spike_height

    return generated_blocks

start_time_old = time()
generate_blocks(num_features, number_blocks, length_blocks, block_height, spike_height, levels, spikes)
end_time_old = time()

print(end_time_old-start_time_old)





from numba import jit

@jit(nopython=True)
def generate_blocks(num_features, number_blocks, length_blocks, block_height,
                    spike_height, levels=False, spikes=0):
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

    generated_blocks = np.zeros(num_features)
    max_blocks = np.floor(num_features / length_blocks)


    start_blocks = np.random.choice(np.arange(max_blocks), number_blocks)


#    if max_blocks < number_blocks:
#        break

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
        non_blocks = np.arange(num_features-number_blocks * length_blocks - spikes)
        for i in range(num_features):
            if generated_blocks[i] == 0:
                for j in range(len(non_blocks)):
                    non_blocks[j] = i
        beta_spikes = np.random.choice(non_blocks, spikes)
        for k in range(len(beta_spikes)):
            beta_spikes[k] = int(beta_spikes[k])
        for i in beta_spikes:
            generated_blocks[i] = spike_height

    return generated_blocks
