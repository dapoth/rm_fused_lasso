from sklearn.base import BaseEstimator, RegressorMixin
import cvxpy as cp
import numpy as np
import math
import random as rd
from sklearn.model_selection import GridSearchCV
import inspect
import matplotlib.pyplot as plt

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
    self.beta = beta
    def __init__( self, s1, s2):
        """ Called when initializing the Fused Lasso Estimator. """
        self.s1 = s1
        self.s2 = s2


    def fit( self, X, y):
        """
        The code for the fused lasso estimator. The penalties s1 and s2 are
        included as additional restriction equations.
        Examples
    --------
    >>> reg = FusedLassoEstimator( s1 = 1, s2 = 10).fit(X, y)

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """

#        if X.shape[0] != y.shape[0]:
#            raise ValueError( "X and y have inconsistent dimensions (%d != %d)"
#                             % (X.shape[0], y.shape[0]))


        p = len(X[1,:])
        b = cp.Variable(p)
        error = cp.sum_squares(X*b - y)
        obj = cp.Minimize(error)
        constraints = [cp.norm(b,1) <= self.s1, cp.norm(b[1:p]-b[0:p-1],1) <= self.s2]
        prob = cp.Problem(obj, constraints)
        prob.solve()

        self.beta = b.value

        return b.value



    def predict( self, X, y = None):
        #try:
        #    getattr(self, "treshold_")
        #except AttributeError:
        #    raise RuntimeError("You must train classifer before predicting data!")

        return np.matmul(X, self.beta)

def generate_blocks(p, number_blocks, length_blocks, amplitude,  spike_level, levels = False, spikes = 0):
    """
    generate beta's for simulation purpose.
    """

    container = np.zeros(p)
    max_blocks = math.floor(p/ length_blocks)

    #blocks = np.linspace(1, number_blocks, number_blocks)
    start_blocks = rd.sample(range(max_blocks),number_blocks)

#    if max_blocks < number_blocks:
#        break

    amplitudes = [amplitude, amplitude*2]

    if levels == True :
        """
        If the Blocks should not all have equal levels, we will randomly chose
        the level of each block as either amplitude or amplitude times 2.
        """

        for block in start_blocks :
            amp = rd.choice(amplitudes)
            for i in range(p) :
                if ( i > ( block-1)* length_blocks) and ( i <= block* length_blocks):
                    container [i] = amp

    else:
        for block in start_blocks :
            #amp = rd.choice(amplitudes)
            for i in range(p) :
                if ( i > ( block-1)* length_blocks) and ( i <= block* length_blocks):
                    container [i] = amplitude

    #if spikes != 0 :
     #   rd.choice(container[:]==0, spikes) = spike_level

    return container
