#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
from src.data_management.generate_data import generate_beta
from src.data_management.generate_data import generate_data
import pytest
import sys

@pytest.fixture
def setup_beta_generation():
    out = {}
    out['num_features'] = 10
    out['number_blocks'] = 2
    out['length_blocks'] = 3
    out['block_height'] = 3
    return out
                      
@pytest.fixture
def setup_data_generation():
    out = {}
    out['num_simulations'] = 100
    out['num_observations'] = 20
    out['num_features'] = 10
    out['num_blocks'] = 2
    out['length_blocks'] = 3
    out['height'] = 3
    out['levels'] = True
    out['spikes'] = 3
    out['spike_height'] = 5
    return out


@pytest.fixture
def expected_output():
    out = {}
    out['blocks'] = 6
    out['blocks_with_spikes'] = 8
    return out

@pytest.fixture
def expected_dimensions():
    out = {}
    out['beta'] = (10,100)
    out['X'] = (20,10)
    out['epsilon'] = (20, 100)
    out['y'] = (20, 100)
    out_dimensions = [out['beta'], out['X'], out['epsilon'], out['y']]
    return out_dimensions
    
def test_generate_beta(setup_beta_generation, expected_output):
    beta_hat = generate_beta(**setup_beta_generation)
    number_of_nonzeros = numpy.count_nonzero(beta_hat)
    expected_nonzeros = expected_output['blocks']
    numpy.testing.assert_array_equal(expected_nonzeros, number_of_nonzeros)
    
def test_generate_beta_with_levels(setup_beta_generation, expected_output):
    setup_beta_generation['levels'] = True
    beta_hat = generate_beta(**setup_beta_generation)
    number_of_nonzeros = numpy.count_nonzero(beta_hat)
    expected_nonzeros = expected_output['blocks']
    numpy.testing.assert_array_equal(expected_nonzeros, number_of_nonzeros)
    
def test_generate_beta_with_spikes(setup_beta_generation, expected_output):
    setup_beta_generation['spikes'] = 2
    beta_hat = generate_beta(**setup_beta_generation)
    number_of_nonzeros = numpy.count_nonzero(beta_hat)
    expected_nonzeros = expected_output['blocks_with_spikes']
    numpy.testing.assert_array_equal(expected_nonzeros, number_of_nonzeros)

def test_generate_too_many_blocks(setup_beta_generation):
    setup_beta_generation['number_blocks'] = 4
    with pytest.raises(TypeError):
        generate_beta(**setup_beta_generation)
        
def test_dimensionality_output_data_generation(setup_data_generation, expected_dimensions):
    beta, X, epsilon, y = generate_data(**setup_data_generation)
    dim_beta = beta.shape
    dim_X = X.shape
    dim_epsilon = epsilon.shape
    dim_y = y.shape
    dimension = [dim_beta, dim_X, dim_epsilon, dim_y]
    numpy.testing.assert_array_equal(dimension, expected_dimensions)
    


if __name__== '__main__':
    status = pytest.main([sys.argv[1]])
    sys.exit(status)
