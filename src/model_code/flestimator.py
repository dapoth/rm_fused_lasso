"""Class that implements the fused lasso estimator.

Class was written as an sklearn.base extension. Notation was therefore adapted
from sklearn. To solve the convex problem we resport to CVXPY.
"""
import cvxpy as cp
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class FusedLassoEstimator(BaseEstimator, RegressorMixin):
    """Define Fused Lasso estimator as extension of Scikit basis class.

    The Fused Lasso Estimator makes a penalized least squares regression
    where diffferences between neighboring betas and non-zero betas are
    penalized.
    The optimization objective for Fused Lasso is:
        (1 / (2 * n_samples)) * ||y - Xb||^2_2
        s.t.        ||b||_1     <= s1
             sum(|b_i-b_{i-1]|) <= s2
    Technically this is a Convex optimization problem that is solved by cvxpy.
    """

    def __init__(self, s1, s2):
        """Call when initializing the Fused Lasso Estimator."""
        self.s1 = s1
        self.s2 = s2


    def fit(self, y, X):
        """Fit unkown parameters *beta* to *X* and *y*.

        Examples:
        --------
        >>> reg = FusedLassoEstimator( s1 = 1, s2 = 10).fit(X, y)

        Args:
            X (np.ndarray): n x p dimensional matrix of independent variables.
            y (np.ndarray): n dimensional vector of dependent variables.

        Returns:
            b.value (np.ndarray)

        """
        if len(y) != len(X):
            raise TypeError("The length of y must be equal to the number of rows of x.")

        if np.size(self.s1) > 1 or np.size(self.s2) > 1:
            raise TypeError("The penalty constants need to have length one.")

        if self.s1 < 0 or self.s2 < 0:
            raise ValueError("The penalty constants need to be nonnnegative.")

        n_features = len(X[1, :])
        beta_hat = cp.Variable(n_features)
        error = cp.sum_squares(X*beta_hat - y)
        obj = cp.Minimize(error)
        constraints = [cp.norm(beta_hat, 1) <= self.s1,
                       cp.norm(beta_hat[1:n_features] - beta_hat[0:n_features-1], 1) <= self.s2]
        prob = cp.Problem(obj, constraints)
        prob.solve()
        self.beta = beta_hat.value

        return beta_hat.value

    def predict(self, X):
        """Predict dependent variable from estimated parameters *beta*.

        Args:
            X (np.ndarray): n x p dimensional matrix of independent variables.
            beta (np.ndarray): p dimensional vector of regression parametes.

        Returns:
            y_hat (np.ndarray)

        """
        #   try:
        #       getattr(self, "treshold_")
        #   except AttributeError:
        #       raise RuntimeError("You must train classifer
        #             before predicting data!")
        y_hat = np.matmul(X, self.beta)

        return y_hat
