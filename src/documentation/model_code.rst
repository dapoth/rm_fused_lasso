.. _model_code:

**********
Model code
**********


The directory *src.model_code* contains source files that are potentially used at various steps of the analysis.

The estimators are used at various steps including the :ref:`analysis` and the :ref:`final` steps.
    
The implementation of the fused lasso :cite:`fused` includes the implementation of the lasso :cite:`lasso` and fusion estimator :cite:`land1996variable` by appropriately chosen penalty constants. The estimator is implemented both as a solution to the lagrange function and as a solution to the primal problem as there are settings, where one prefers one of the two. In order to perform gridcv there is a third implementation of the estimator, in which....


``Fused lasso estimator of the lagrange problem``
=================================================

.. automodule:: src.model_code.fused_lasso_dual
    :members:
    
    
``Fused lasso estimator of the primal problem``
===============================================

.. automodule:: src.model_code.fused_lasso_primal
    :members:


``flestimator``
================

.. automodule:: src.model_code.fused_lasso_primal
    :members:
    

``Tests for the fused lasso``
==============================

.. automodule:: src.model_code.test_fused_lasso
    :members:
