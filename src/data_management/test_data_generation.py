#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 23:30:29 2019

@author: clara
"""

import numpy
from src.data_management.generate_data import generate_beta
import pytest

@pytest.fixture
def setup_block_generation():
    out = {}
    out['num_features'] = 10
    out['number_blocks'] = 3
    out['length_blocks'] = 3
    out['block_height'] = 3
    out['spike_height'] = 0
    return out

@pytest.fixture
def expected_output():
    out = {}
    out['nonzeros'] = 9
    return out

def test_generate_beta(setup_block_generation):
    beta_hat = generate_beta(**setup_block_generation)
    number_of_nonzeros = numpy.count_nonzero(beta_hat)
    numpy.testing.assert_array_equal(9, number_of_nonzeros)

def test_generate_too_many_blocks(setup_block_generation):
    setup_block_generation['number_blocks'] = 4
    with pytest.raises(TypeError):
        generate_beta(**setup_block_generation)
