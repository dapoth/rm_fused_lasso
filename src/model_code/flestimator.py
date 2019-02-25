import cvxpy as cp
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class FusedLassoEstimator(BaseEstimator, RegressorMixin):
    """
    Fused Lasso Estimator that makes a penalized least squares regression
    where diffferences between neighboring betas and non-zero betas get
    penalized.
    The optimization objective for Fused Lasso is:
        (1 / (2 * n_samples)) * ||y - Xb||^2_2
        s.t.        ||b||_1     <= s1
             sum(|b_i-b_{i-1]|) <= s2
    Technically this is a Convex optimization problem that is solved by cvxpy.
    """
    def __init__(self, s1, s2):
        """ Called when initializing the Fused Lasso Estimator. """
        self.s1 = s1
        self.s2 = s2

    def fit(self, X, y=None):
        """
        The code for the fused lasso estimator. The penalties s1 and s2 are
        included as additional restriction equations.
        Examples
        --------
        >>> reg = FusedLassoEstimator( s1 = 1, s2 = 10).fit(X, y)

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """
        p = len(X[1, :])
        b = cp.Variable(p)
        error = cp.sum_squares(X*b - y)
        obj = cp.Minimize(error)
        constraints = [cp.norm(b, 1) <= self.s1, cp.norm(b[1:p]-b[0:p-1], 1)
                       <= self.s2]
        prob = cp.Problem(obj, constraints)
        prob.solve()
        self.beta = b.value

        return b.value

    def predict(self, X, beta=None):
        """Docstring."""
        #   try:
        #       getattr(self, "treshold_")
        #   except AttributeError:
        #       raise RuntimeError("You must train classifer
        #             before predicting data!")

        return np.matmul(X, self.beta)
