.. _model_code:

**********
Model code
**********


The directory *src.model_code* contains the implementation of our estimators.

The estimators are used at the :ref:`analysis` step.
    
The implementation of the fused lasso :cite:`fused` includes the implementation of the lasso :cite:`lasso` and fusion estimator :cite:`land1996variable` by appropriately chosen penalty constants.
One obtains the lasso by setting s2 extremely large (for the primal problem) and by setting  lambda1 extremely small (for the lagrange function). Note that setting lambda2 = 0 might cause problems with the optimizer, so rather set lambda2 = 0.000001. Similarly the fusion estimator is obtained by setting s1 extremely large (for the primal problem) and by setting  lambda1 extremely small (for the lagrange function).
The estimator is implemented both as a solution to the lagrange function and as a solution to the primal problem as there are settings, where one prefers one of the two. In order to perform GridCV there is a third implementation of the estimator, in which the estimator is a class written as an sklearn.base extension (see https://scikit-learn.org/stable/modules/classes.html#module-sklearn.base for details.)


Fused lasso estimator of the lagrange problem
==============================================

.. automodule:: src.model_code.fused_lasso_lagrange
    :members:
    
    
Fused lasso estimator of the primal problem
============================================

.. automodule:: src.model_code.fused_lasso_primal
    :members:


Fused lasso estimator as a class
=================================

.. automodule:: src.model_code.flestimator
    :members:

Tests
=====

We perform unit tests for the three estimators. We check that the estimators compute the correct results in a setting, where *X* is the identity matrix and where *X* is  not a diagonal matrix. Expected test results were computed either by hand (for the case where *X* is the identity matrix) or with the help of the Penalized package for R (see :cite:`goeman2018l1`) in the case, where *X* is not the identity matrix. 
As the flestimator calls the fused_lasso_primal function we checked that it does so correctly in a test. Furthermore we checked that the functions raise an error, when inappropriate inputs are passed.

.. automodule:: src.model_code.test_fused_lasso
    :members:
