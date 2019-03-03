import numpy as np
import random as rd
import math

def generate_blocks(p, number_blocks, length_blocks, amplitude,
                    spike_level, levels=False, spikes=0):
    """
    Generate beta's for simulation purpose.
    """

    container = np.zeros(p)
    max_blocks = math.floor(p / length_blocks)

    start_blocks = rd.sample(range(max_blocks), number_blocks)

#    if max_blocks < number_blocks:
#        break

    if levels:
        """
        If the Blocks should not all have equal levels, we will randomly chose
        the level of each block as either amplitude or amplitude times 2.
        """
        amplitudes = [amplitude, amplitude*2]
        for block in start_blocks:
            amp = rd.choice(amplitudes)
            for i in range(p):
                if (i >= block * length_blocks) and (i < (block+1) *
                                                     length_blocks):
                    container[i] = amp

    else:
        for block in start_blocks:
            for i in range(p):
                if (i >= block * length_blocks) and (i < (block+1) *
                                                     length_blocks):
                    container[i] = amplitude

    if spikes != 0:
        non_blocks = []
        for i in range(p):
            if container[i] == 0:
                non_blocks.append(i)
        beta_spikes = rd.sample(non_blocks, spikes)
        for i in beta_spikes:
            container[i] = spike_level

    return container
