"""Plot the solution path of the lasso estimator."""
import numpy as np
from src.model_code.lasso_solution_path import lasso_solution_path
from bld.project_paths import project_paths_join as ppj

# Initialize setting.
N_FEATURES = 10
N_OBS = 10
BETA = np.zeros(N_FEATURES)
BETA[3:6] = 1
np.random.seed(1000)
MEAN = np.zeros(N_FEATURES)
COV = np.identity(N_FEATURES)
X = np.random.multivariate_normal(MEAN, COV, N_OBS)
EPSILON = np.random.randn(N_OBS)
Y = np.matmul(X, BETA) + EPSILON

# Calculate solution path and save plot.
PLOT = lasso_solution_path(Y, X)
PLOT.savefig(ppj("OUT_FIGURES", "plot_solutionpath_lasso.png"))
PLOT.clf()
