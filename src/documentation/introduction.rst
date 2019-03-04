.. _introduction:


************
Introduction
************

Documentation on the rationale, Waf, and more background is at http://hmgaudecker.github.io/econ-project-templates/

.. _getting_started:

Getting started
===============

**This assumes you have completed the steps in the** `README.md file <https://github.com/hmgaudecker/econ-project-templates/>`_ **and everything worked.**

The logic of the project template works by step of the analysis: 

1. Data management
2. The actual estimations / simulations / ?
3. Visualisation and results formatting (e.g. exporting of LaTeX tables)
4. Research paper and presentations. 

It can be useful to have code and model parameters available to more than one of these steps, in that case see sections :ref:`model_specifications`, :ref:`model_code`, and :ref:`library`.


Idea
=====

This project aims at implementing the one-dimensional fused lasso in Python and applying it to simulated datasets and a genomic dataset. Also the fused lasso and lasso are studied from a theoretical point of view in a paper. The paper also examines the performance of the lasso, fusion estimator and fused lasso on the simulated datasets.


The code is unit tested.

The directory is structured as follows. First we call data_management. Here the data for the simulation is generated and stored in bld.out.data. Secondly we perform a simulation study and Monte Carlo simulation with the estimators defined in src.model_code. With the results we create plots and tables that are constructed in :ref:`final` and saved under bld.out.figures. They are also included in the paper, which is created in :ref:`paper`.
