.. _data_management:

***************
Data management
***************


Documentation of the code in *src.data_management*. In this section are all the relevant codes for the generation of the simulation data. From the JSON files in :ref:`model_specs` all the relevant model parameters are extracted to generate four different datasets.

1. Large_blocks
2. Small_blocks
3. Blocks_few_spikes
4. Spikes

These settings were chosen to demonstrate the performance of the fused lasso in settings where it is more or less appropriate.


Generate data for one simulation
=================================

.. automodule:: src.data_management.generate_blocks
    :members:

Generate data for all simulations
==================================

.. automodule:: src.data_management.generate_data
    :members:
    
