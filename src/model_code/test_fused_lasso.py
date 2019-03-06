#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numpy.testing import assert_allclose
import pytest
from src.model_code.fused_lasso_dual import fused_lasso_dual
from src.model_code.fused_lasso_primal import fused_lasso_primal
from src.model_code.flestimator import FusedLassoEstimator as fle


@pytest.fixture
def setup_param_lasso():
    out_dual = {}
    out_dual['y'] = [0.25, 3]
    out_dual['X'] = np.identity(2)
    out_dual['lambda1'] = 1
    out_dual['lambda2'] = 0

    out_primal = {}
    out_primal['y'] = [0.25, 3]
    out_primal['X'] = np.identity(2)
    out_primal['s1'] = 2.5
    out_primal['s2'] = 100000

    out = {"data_dual": out_dual, "data_primal": out_primal}
    return out


@pytest.fixture
def setup_param_fused_lasso():
    out_dual = {}
    out_dual['y'] = [0.1, 3, 3]
    out_dual['X'] = np.identity(3)
    out_dual['lambda1'] = 1
    out_dual['lambda2'] = 1

    out_primal = {}
    out_primal['y'] = [0.1, 3, 3]
    out_primal['X'] = np.identity(3)
    out_primal['s1'] = 4.600003
    out_primal['s2'] = 2.149997

    out = {"data_dual": out_dual, "data_primal": out_primal}
    return out

@pytest.fixture
def setup_param_flestimator():
    out_penalties = {}
    out_penalties['s1'] = 4.600003
    out_penalties['s2'] = 2.149997
    
    out_data = {}
    out_data['y'] = [0.1, 3, 3]
    out_data['X'] = np.identity(3)
    
    out = {"penalties": out_penalties, "data": out_data}
    return out

@pytest.fixture
def expected_beta():
    out_signal = {}
    out_signal['lasso'] = [0, 2.5]
    out_signal['fused_lasso'] = [0.100003, 2.25, 2.25]

    out_general = {}
    out_general['lasso'] = [0, 2.09999]
    out_general['fused_lasso'] = [0, 1.09524, 2.3619]

    out = {"values_signal": out_signal, "values_general": out_general}
    return out


# Test the functionality of the lasso estimator, which one receives by setting
# lambda2=0 or s2 to a very high number.
def test_lasso_signal_dual(setup_param_lasso, expected_beta):
    expected_beta_lasso_signal = expected_beta["values_signal"]["lasso"]
    calculated_beta_dual = fused_lasso_dual(**setup_param_lasso["data_dual"])
    assert_allclose(expected_beta_lasso_signal, calculated_beta_dual, atol=1e-2)


def test_lasso_signal_primal(setup_param_lasso, expected_beta):
    expected_beta_lasso_signal = expected_beta["values_signal"]["lasso"]
    calculated_beta_primal = fused_lasso_primal(**setup_param_lasso["data_"
                                                                    "primal"])
    assert_allclose(expected_beta_lasso_signal, calculated_beta_primal,
                    atol=1e-2)


def test_lasso_general_dual(setup_param_lasso, expected_beta):
    setup_param_lasso["data_dual"]['X'] = np.array([[1, 0.5], [0.5, 1]])
    expected_beta_lasso_general = expected_beta["values_general"]["lasso"]
    calculated_beta_dual = fused_lasso_dual(**setup_param_lasso["data_dual"])
    assert_allclose(expected_beta_lasso_general, calculated_beta_dual,
                    atol=1e-2)


def test_lasso_general_primal(setup_param_lasso, expected_beta):
    setup_param_lasso["data_primal"]['X'] = np.array([[1, 0.5], [0.5, 1]])
    setup_param_lasso["data_primal"]['s1'] = 2.09999
    expected_beta_lasso_general = expected_beta["values_general"]['lasso']
    calculated_beta_dual = fused_lasso_primal(**setup_param_lasso["data_primal"])
    assert_allclose(expected_beta_lasso_general, calculated_beta_dual, atol=1e-2)


# Test the functionality of the fused lasso estimator.
def test_fused_lasso_signal_dual(setup_param_fused_lasso, expected_beta):
    expected_beta_fused_lasso_signal = expected_beta["values_signal"]['fused_'
                                                                      'lasso']
    calculated_beta_dual = fused_lasso_dual(**setup_param_fused_lasso["data_"
                                                                      "dual"])
    assert_allclose(expected_beta_fused_lasso_signal, calculated_beta_dual,
                    atol=1e-2)


def test_fused_lasso_general_dual(setup_param_fused_lasso, expected_beta):
    expected_beta_fused_lasso_general = expected_beta["values_general"]['fused_'
                                                                        'lasso']
    setup_param_fused_lasso["data_dual"]['X'] = np.array([[1, 0.5, 0], [0, 1,
                                                         0.5], [0.5, 0, 1]])
    calculated_beta_dual = fused_lasso_dual(**setup_param_fused_lasso["data_"
                                                                      "dual"])
    assert_allclose(expected_beta_fused_lasso_general, calculated_beta_dual,
                    atol=1e-2)


# Test the functionality of the flestimator.
def test_flestimator(setup_param_flestimator, expected_beta):
    clf = fle(**setup_param_flestimator["penalties"])
    beta_hat = clf.fit(**setup_param_flestimator["data"])
    expected_beta_fused_lasso_signal = expected_beta["values_signal"]['fused_'
                                                                      'lasso']
    assert_allclose(expected_beta_fused_lasso_signal, beta_hat,
                    atol=1e-2)
    

# Test that the function does not work with incorrect input.
def test_fused_lasso_dual_negative_penalty(setup_param_lasso):
    setup_param_lasso["data_dual"]['lambda1'] = -1
    with pytest.raises(ValueError):
        fused_lasso_dual(**setup_param_lasso["data_dual"])


def test_fused_lasso_dual_wrong_dimensions(setup_param_lasso):
    setup_param_lasso["data_dual"]['y'] = [1, 2, 3]
    with pytest.raises(TypeError):
        fused_lasso_dual(**setup_param_lasso["data_dual"])


def test_fused_lasso_dual_wrong_dimension_penalty(setup_param_lasso):
    setup_param_lasso["data_dual"]['lambda1'] = [1, 2]
    with pytest.raises(TypeError):
        fused_lasso_dual(**setup_param_lasso["data_dual"])

def test_fused_lasso_primal_negative_penalty(setup_param_lasso):
    setup_param_lasso["data_primal"]['s1'] = -1
    with pytest.raises(ValueError):
        fused_lasso_primal(**setup_param_lasso["data_primal"])


def test_fused_lasso_primal_wrong_dimensions(setup_param_lasso):
    setup_param_lasso["data_primal"]['y'] = [1, 2, 3]
    with pytest.raises(TypeError):
        fused_lasso_primal(**setup_param_lasso["data_primal"])


def test_fused_lasso_primal_wrong_dimension_penalty(setup_param_lasso):
    setup_param_lasso["data_primal"]['s1'] = [1, 2]
    with pytest.raises(TypeError):
        fused_lasso_primal(**setup_param_lasso["data_primal"])

