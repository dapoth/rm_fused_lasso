.. _final:

************************************
Visualisation and results formatting
************************************


Documentation of the code in *src.final*. The folder generates the plots and the table of the results of the simulation.


CGH Data
=================

Apply the fused lasso estimator to the CGH data contained in :ref:`original_data` and plot it.

.. automodule:: src.final.cgh_plot
    :members:

Table with the results from the simulation
===========================================

The table reports the mean squared error of the corresponding estimator, the specificity (proportion of true zeros detected), the sensitivity (proportion of true non-zeros detected) and the percentage of blocks detected as well as standard errors.

.. automodule:: src.final.create_analysis_table
    :members:
    
Create solution paths for lasso and fused lasso
================================================

.. automodule:: src.final.fused_lasso_solution_path_plot
    :members:

.. automodule:: src.final.lasso_solution_path_plot
    :members:
