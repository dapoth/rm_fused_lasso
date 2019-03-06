.. _analysis:

************************************
Main model simulations
************************************

Documentation of the code in *src.analysis*. This is the core of the project. It contains a simulation study and a Monte Carlo simulation.


Fused Lasso example
===================

At first the optimal penalty constants s1 and s2 are computed via cross-validation over a two-dimensional grid. Then the three estimators lasso, fusion and fused lasso are computed for the four different settings described in :ref:`data_management`.

.. automodule:: src.analysis.simulation
    :members:

    
Monte Carlo
============

For the setting Large_blocks a Monte Carlo simulation is conducted. The distribution of four different specific coefficients is analysed.
A coefficient:

* in the center of a block
* at the boundary of a block
* which is zero and not next to a block
* which is zero, but next to a block.
    
.. automodule:: src.analysis.monte_carlo
    :members:


Analysis of simulation
=======================

.. automodule:: src.analysis.analysis_of_simulations
    :members:
