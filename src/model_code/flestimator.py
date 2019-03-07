"""Class that implements the fused lasso estimator.

The flestimator was written as an sklearn.base extension. Notation is therefore
adapted to sklearn. To solve the convex problem we resort to CVXPY.

"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from src.model_code.fused_lasso_primal import fused_lasso_primal


class FusedLassoEstimator(BaseEstimator, RegressorMixin):
    """Define fused lasso estimator as extension of Scikit basis class.

    The fused lasso estimator makes a penalized least squares regression,
    where non-zero betas and diffferences between neighboring coefficients are
    penalized.
    The optimization objective for fused lasso is:
        ||y - Xb||^2_2
        s.t.        ||b||_1     <= s1
             sum(|b_i-b_{i-1]|) <= s2
    This is a convex optimization problem that is solved by CVXPY.
    """

    def __init__(self, s1, s2):
        """Call when initializing the fused lasso estimator."""
        self.s1 = s1
        self.s2 = s2


    def fit(self, X, y):
        """Fit unkown parameters *beta_hat* to *X* and *y*.

        Example:
        --------
        >>> reg = FusedLassoEstimator( s1=1, s2=10).fit(X, y)

        Args:
            | X (np.ndarray): 2d array of independent variables.
            | y (np.ndarray): 1d array of dependent variables.

        Returns:
            | b.value (np.ndarray)

        """
        self.beta = fused_lasso_primal(y, X, self.s1, self.s2)

        return self.beta

    def predict(self, X):
        """Predict dependent variable from estimated parameters *beta_hat*.

        Args:
            | X (np.ndarray): 2d array of independent variables.
            | beta (np.ndarray): 1d array of regression parametes.

        Returns:
            | y_hat (np.ndarray)

        """
        y_hat = np.matmul(X, self.beta)

        return y_hat
