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

import cvxpy as cp

def fused_lasso_dual(y, X, lambda1, lambda2):
    """Solves for given data and penalty constants the fused lasso dual form.

    Args:
        y (np.ndarray): 1d array of dependent variables
        X (np.ndarray): 2d array of independent variables
        s1 (float): constraint on ||b||_1
        s2 (float): constraint on the absolute jumps in beta

    Returns:
        beta.value (np.ndarray)

    """
    if len(y) != len(X):
        raise ValueError("The length of y must be equal to the number of rows of x.")
    #if lambda1 < 0 | lambda2 < 0:
    #    raise ValueError("The penalty constraints need to be nonnnegative.")
    n = len(X)
    n_features = len(X[1, :])
    beta = cp.Variable(n_features)
    error = cp.sum_squares(X*beta - y)
    obj = cp.Minimize( (1 / (2 * n )) * error + lambda1 * cp.norm(beta, 1) + lambda2*
                      cp.norm(beta[1:n_features]-beta[0:n_features-1], 1))
    prob = cp.Problem(obj)
    prob.solve()

    return beta.value

import numpy as np
from sklearn import linear_model
clf = linear_model.Lasso(alpha=1,fit_intercept=False)
clf.fit(X,[0.25,3])
clf.coef_

fused_lasso_dual([0.25, 3], X, 1, 0)


X = np.random.multivariate_normal(np.zeros(2), np.identity(2), 2)
