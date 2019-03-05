# rm_fused_lasso
## Implementation and application of the fused lasso.
### Introduction
This repository implements the fused lasso (and thereby the fusion estimator and the lasso) in python using a convex solver and then applies the estimators to simuated data and a real dataset.
### Installation
The repository is structured according to the Waf template by Hans-Martin von Gaudecker which you can find [here](https://github.com/hmgaudecker/econ-project-templates). To run the code you need to have Conda. In case you use Anaconda or Miniconda you have it already.
The execution of the code further requires the [CVXPY](https://www.cvxpy.org/) software package. For installation instructions see [here](https://www.cvxpy.org/install/index.html). 
### Execution
Once everything is installed, clone the repository and open a shell in its main directory. Then you need to type the following commands:

1. ``python waf.py configure``
2. ``python waf.py build``

The build step will take a few minutes as a lot of estimations are performed during the cross-validations.
Once the step is completed, you will find plots and the table of simulation results in the bld directory. Also the estimators are stored in the bld directory.
