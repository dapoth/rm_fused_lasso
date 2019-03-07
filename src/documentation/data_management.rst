.. _data_management:

***************
Data management
***************


Documentation of the code in *src.data_management*. In this section are all the codes for the generation of the simulation data. From the JSON files in :ref:`model_specifications` all the relevant model parameters are extracted to generate four different datasets each consisting itself of 'number of simulations' many datasets. Each of these datasets consists of a vector of true coefficients *beta*, a design matrix *X*, a vector of error terms *epsilon* and a vector of observations *y*. A linear model is supposed. The four settings only differ by their coefficient vector *beta* of the 40 regressors. By a block we mean neighboring regressors that have the same non-zero value, a spike is a single nonzero regressor.

1. **Large_blocks**: 3 blocks of length 10 with height 3
2. **Small_blocks**: 10 blocks of length 3 with height 3
3. **Blocks_few_spikes**: 9 blocks of length 3 with height 3 and 3 spikes of height 7
4. **Spikes**: 20 spikes of height 7

These settings were chosen to demonstrate the performance of the fused lasso in settings where it is more or less appropriate. The block settings are the settings for which it was designed.

Generate data for all simulations
==================================

.. automodule:: src.data_management.functions_for_data_generation
    :members:


Tests
======

We perform tests for the generate_beta function and the generate_data function. For the generate_beta function we test that it produces the right amount of nonzero coefficients and raises an error, when one tries to run the function with invalid input (there are too many blocks for the number of regressors). Further we test that the dimensions of the output of generate_data are correct, so we have valid datasets for the analysis.

.. automodule:: src.data_management.test_functions_for_data_generation
    :members:
