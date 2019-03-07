.. _analysis:

************************************
Main model simulations
************************************

Documentation of the code in *src.analysis*. It contains a simulation study and a Monte Carlo simulation.


Fused lasso applied to simulated datasets
========================================== 

.. automodule:: src.analysis.grid_cross_validation
    :members:


.. automodule:: src.analysis.estimation
    :members:


Monte Carlo
============

For the setting **Large_blocks** (see :ref:`data_management`) a Monte Carlo simulation is conducted. The distribution of four different specific coefficients is analysed and plotted in a histogram.
A coefficient:

* in the center of a block
* at the boundary of a block
* which is zero and not next to a block
* which is zero, but next to a block.


Analysis of simulation
=======================

.. automodule:: src.analysis.analysis_of_simulations
    :members:
