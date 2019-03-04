.. _model_code:

**********
Model code
**********


The directory *src.model_code* contains source files that might differ by model and that are potentially used at various steps of the analysis.

For example, you may have a class that is used both in the :ref:`analysis` and the :ref:`final` steps. Additionally, maybe you have different utility functions in the baseline version and for your robustness check. You can just inherit from the baseline class and override the utility function then.
    
    
The implementation of the fused lasso :cite:`fused` includes the implementation of the lasso :cite:`lasso` and fusion estimator :cite:`land1996variable` for appropriate penalty constants.


``Fused lasso estimator of the lagrange problem``
=============================================

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
