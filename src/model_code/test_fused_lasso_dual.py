#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 11:10:05 2019

@author: clara
"""

import numpy as np
from fused_lasso_dual import fused_lasso_dual
from numpy.testing import assert_allclose
import pytest

expected_beta = [0, 2.5]

@pytest.fixture
def setup_param():
    out = {}
    out['y'] = [0.25, 3]
    out['x'] = np.identity(2)
    out['lambda1'] = 1
    out['lamda2'] = 0
    
def test_fused_lasso_dual(setup_param):
    calculated_beta = fused_lasso_dual(**setup_param)
    assert_allclose(expected_beta, calculated_beta)

#test_fused_lasso_dual([0.25,3], np.identity(2), 1, 0)