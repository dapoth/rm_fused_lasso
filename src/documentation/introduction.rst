.. _introduction:


************
Introduction
************


Aim
====

This project aims at implementing the one-dimensional fused lasso in Python and applying it to simulated datasets and a genomic dataset. Also the fused lasso and lasso are studied from a theoretical point of view in a paper. The paper further examines the performance of the lasso, fusion estimator and fused lasso on the simulated datasets.

Documentation on the rationale, Waf, and more background is at http://hmgaudecker.github.io/econ-project-templates/.

.. _getting_started:

Getting started
===============

This assumes you have completed the installation instructions in the `README.md <https://github.com/dapoth/rm_fused_lasso/blob/master/README.md>`_ of the project.
The used `template <https://github.com/hmgaudecker/econ-project-templates>`_ ensures that the directory is split into logical steps of the analysis as described now. 

.. _structure:

Structure
==========

First we call data_management. Here the data for the simulation is generated and stored in bld.out.data as `pickle files <https://docs.python.org/3/library/pickle.html>`_. It is generated according to different model settings stored in :ref:`model_specifications`. Secondly, in :ref:`model_analysis` we perform a simulation study and Monte Carlo simulation with the estimators we implemented in :ref:`model_code`. The implementation of the estimators including the appropriate choice of a penalty constant via two-dimensional cross-validation are the main achievement of this repository. 

The estimators as well as the data generation are tested by means of unit tests.

After estimation we analyze the results in :ref:`analysis`. On the basis of this analysis plots and a table are constructed in :ref:`final` and saved under bld.out.figures. They are also included in the paper, which is created in :ref:`paper`.
