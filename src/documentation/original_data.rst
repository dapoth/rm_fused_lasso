.. _original_data:

*************
Original data
*************


Comparative Genomic Hybridization Data (CGH data)
=================================================

The dataset was taken from https://web.stanford.edu/~hastie/StatLearnSparsity/data.html (the website was visited on 9th Decmber 2018). It contains 990 observations and for each observation the log 2 ratio of the copy number of the gene in the tumor cell in comparison to the healthy cells.


Background CGH
==============

Comparative Genomic Hybridization (CGH) is an increasingly popular method for the molecular analysis of cancer. CGH provides an overview of changes in DNA sequence copy numbers in a tumor sample relative to a healthy control sample. Changes can be in form of losses, deletions, gains and amplifications \citep{cghmain}. The idea is that in cancer cells mutations can cause these changes . Biological knowledge suggest that it is typically segments of a chromosome, that means a couple of neighbouring genes, that are replicated and not single genes.

The aim of the analysis of the data is to detect these regions of gains and losses of copy numbers.
In the CGH experiment DNA sequence copy numbers are measured for selected genes on the chromosome. Unfortunately the data are very noisy and render some kind of smoothing necessary before being analyzable. The fused lasso signal approximator, which is studied in the :ref:`paper`, is an appropriate method for this type of smoothing.
