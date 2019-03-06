#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
from src.data_management.generate_data import generate_beta
import pytest
import sys

@pytest.fixture
def setup_block_generation():
    out = {}
    out['num_features'] = 10
    out['number_blocks'] = 2
    out['length_blocks'] = 3
    out['block_height'] = 3
    return out

@pytest.fixture
def expected_output():
    out = {}
    out['blocks'] = 6
    out['blocks_with_spikes'] = 8
    return out
    
def test_generate_beta(setup_block_generation, expected_output):
    beta_hat = generate_beta(**setup_block_generation)
    number_of_nonzeros = numpy.count_nonzero(beta_hat)
    expected_nonzeros = expected_output['blocks']
    numpy.testing.assert_array_equal(expected_nonzeros, number_of_nonzeros)
    
def test_generate_beta_with_levels(setup_block_generation, expected_output):
    setup_block_generation['levels'] = True
    beta_hat = generate_beta(**setup_block_generation)
    number_of_nonzeros = numpy.count_nonzero(beta_hat)
    expected_nonzeros = expected_output['blocks']
    numpy.testing.assert_array_equal(expected_nonzeros, number_of_nonzeros)
    
def test_generate_beta_with_spikes(setup_block_generation, expected_output):
    setup_block_generation['spikes'] = 2
    beta_hat = generate_beta(**setup_block_generation)
    number_of_nonzeros = numpy.count_nonzero(beta_hat)
    expected_nonzeros = expected_output['blocks_with_spikes']
    numpy.testing.assert_array_equal(expected_nonzeros, number_of_nonzeros)

def test_generate_too_many_blocks(setup_block_generation):
    setup_block_generation['number_blocks'] = 4
    with pytest.raises(TypeError):
        generate_beta(**setup_block_generation)


if __name__== '__main__':
    status = pytest.main([sys.argv[1]])
    sys.exit(status)
