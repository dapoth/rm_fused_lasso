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

expected_beta = [0, 2.5]

@pytest.fixture
def setup_param_signal():
    out_dual = {}
    out_dual['y'] = [0.25, 3]
    out_dual['x'] = np.identity(2)
    out_dual['lambda1'] = 1
    out_dual['lambda2'] = 0
    
    out_primal = {}
    out_primal['y'] = [0.25, 3]
    out_primal['x'] = np.identity(2)
    out_primal['s1'] = 2.5
    out_primal['s2'] = 100000
    
    out = {"data_dual": out_dual, "data_primal": out_primal}
    return out

@pytest.fixture
def setup_param():
    out_dual = {}
    out_dual['y'] = [0.25, 3]
    out_dual["x"] = np.array([[[1, 0.5], [0.5, 1]]])
    out_dual['lambda1'] = 1
    out_dual['lambda2'] = 0

    out_primal = {}
    out_primal['y'] = [0.25, 3]
    out_primal['x'] = np.array([[[1, 0.5], [0.5, 1]]])
    out_primal['s1'] = 2.5
    out_primal['s2'] = 100000
    
    out = {"data_dual": out_dual, "data_primal": out_primal}
    return out
    
def test_lasso_signal_dual(setup_param_signal):
    calculated_beta_dual = fused_lasso_dual(**setup_param_signal["data_dual"])
    assert_allclose(expected_beta, calculated_beta_dual, atol=1e-2)
    
def test_lasso_signal_primal(setup_param_signal):
    calculated_beta_primal = fused_lasso_primal(**setup_param_signal["data_primal"])
    assert_allclose(expected_beta, calculated_beta_primal, atol=1e-2)
    

