#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 11:10:05 2019

@author: clara
"""

import numpy as np
from src.model_code.fused_lasso_dual import fused_lasso_dual
from src.model_code.fused_lasso_primal import fused_lasso_primal
from numpy.testing import assert_allclose
import pytest

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
def expected_beta():
    out_signal = {}
    out_signal['expected_beta_lasso'] = [0, 2.5]
    out_signal['expected_beta_fused_lasso'] = [0.100003, 2.25, 2.25]
    
    out_general = {}
    out_general['expected_beta_lasso'] = [0, 2.09999]
    out_general['expected_beta_fused_lasso'] = [0, 1.09524, 2.3619]
    
    out = {"values_signal": out_signal, "values_general": out_general}
    return out


# Test the functionality of the lasso estimator, which one receives by setting 
# lambda2=0 or s2 to a very high number.    
def test_lasso_signal_dual(setup_param_lasso, expected_beta):
    expected_beta_lasso_signal = expected_beta["values_signal"]['expected_beta_lasso']
    calculated_beta_dual = fused_lasso_dual(**setup_param_lasso["data_dual"])
    assert_allclose(expected_beta_lasso_signal, calculated_beta_dual, atol=1e-2)
    
def test_lasso_signal_primal(setup_param_lasso, expected_beta):
    expected_beta_lasso_signal = expected_beta["values_signal"]['expected_beta_lasso']
    calculated_beta_primal = fused_lasso_primal(**setup_param_lasso["data_primal"])
    assert_allclose(expected_beta_lasso_signal, calculated_beta_primal, atol=1e-2)
    
def test_lasso_general_dual(setup_param_lasso, expected_beta):
    setup_param_lasso["data_dual"]['X'] = np.array([[1, 0.5], [0.5, 1]])
    expected_beta_lasso_general = expected_beta["values_general"]['expected_beta_lasso']
    calculated_beta_dual = fused_lasso_dual(**setup_param_lasso["data_dual"])
    assert_allclose(expected_beta_lasso_general, calculated_beta_dual, atol=1e-2)
    
def test_lasso_general_primal(setup_param_lasso, expected_beta):
    setup_param_lasso["data_primal"]['X'] = np.array([[1, 0.5], [0.5, 1]])
    setup_param_lasso["data_primal"]['s1'] = 2.09999
    expected_beta_lasso_general = expected_beta["values_general"]['expected_beta_lasso']
    calculated_beta_dual = fused_lasso_primal(**setup_param_lasso["data_primal"])
    assert_allclose(expected_beta_lasso_general, calculated_beta_dual, atol=1e-2)
    
# Test the functionality of the fused lasso estimator.
def test_fused_lasso_signal_dual(setup_param_fused_lasso, expected_beta):
    expected_beta_fused_lasso_signal = expected_beta["values_signal"]['expected_beta_fused_lasso']
    calculated_beta_dual = fused_lasso_dual(**setup_param_fused_lasso["data_dual"])
    assert_allclose(expected_beta_fused_lasso_signal, calculated_beta_dual, atol=1e-2)
    
def test_fused_lasso_general_dual(setup_param_fused_lasso, expected_beta):
    expected_beta_fused_lasso_general = expected_beta["values_general"]['expected_beta_fused_lasso']
    setup_param_fused_lasso["data_dual"]['X'] = np.array([[1, 0.5, 0], [0, 1, 0.5], [0.5, 0, 1]])
    calculated_beta_dual = fused_lasso_dual(**setup_param_fused_lasso["data_dual"])
    assert_allclose(expected_beta_fused_lasso_general, calculated_beta_dual, atol=1e-2)
    
# =============================================================================
# def test_lasso_signal_primal(setup_param_fused_lasso):
#     calculated_beta_primal = fused_lasso_primal(**setup_param_fused_lasso["data_primal"])
#     assert_allclose(**expected_beta["values_signal"], calculated_beta_primal, atol=1e-2)
# 
# =============================================================================
