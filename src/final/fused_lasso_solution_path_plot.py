"""Plot the solution path of the fused lasso estimator for fixed s1."""
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from bld.project_paths import project_paths_join as ppj
from src.model_code.fused_lasso_solution_path import fused_lasso_solution_path





N_FEATURES = 10
N_OBS = 10
BETA_HAT = np.zeros(N_FEATURES)
BETA_HAT[3:6] = 3

np.random.seed(1000)
MEAN = np.zeros(N_FEATURES)
COV = np.identity(N_FEATURES)
DESIGN_MAT = np.random.multivariate_normal(MEAN, COV, N_OBS)
EPS = np.random.randn(N_OBS)
OUTCOME = np.matmul(DESIGN_MAT, BETA_HAT) + EPS

fused_lasso_solution_path(OUTCOME, DESIGN_MAT)
