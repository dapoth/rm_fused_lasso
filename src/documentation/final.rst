.. _final:

************************************
Visualisation and results formatting
************************************


Documentation of the code in *src.final*. The folder generates the plots and the table of the results of the simulation.


Estimators applied to one dataset of each of the four settings.
================================================================

.. automodule:: src.final.plot_different_estimators
    :members:

Fused lasso signal approximator applied to CGH Data
====================================================

.. automodule:: src.final.cgh_plot
    :members:

Table with the results from the simulation
===========================================

.. automodule:: src.final.create_analysis_table
    :members:
    
The table reports the mean squared error of the corresponding estimator, the specificity (proportion of true zeros detected), the sensitivity (proportion of true non-zeros detected) and the percentage of blocks detected as well as standard errors.
    
Create solution paths for lasso and fused lasso
================================================

.. automodule:: src.final.fused_lasso_solution_path_plot
    :members:

.. automodule:: src.final.lasso_solution_path_plot
    :members:


Create heatmaps for the results of grid cv
===========================================

.. automodule:: src.final.heatmap
    :members:
