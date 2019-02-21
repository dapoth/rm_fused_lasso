#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 08:42:40 2019

@author: christopher
"""

import matplotlib.pyplot as plt
import numpy as np
from bld.project_paths import project_paths_join as ppj
import sys
import pickle


if __name__ == "__main__":

    cgh_data = np.loadtxt(ppj("IN_DATA", "cgh.txt"))




    # Plot normball corresponding to absolute value
    fig = plt.figure()
    plt.scatter([1,0,-1,0,1],[0,1,0,-1,0])
    plt.plot([1,0,-1,0,1],[0,1,0,-1,0])
    plt.fill([1,0,-1,0,1],[0,1,0,-1,0],color='blue', alpha=0.5)

    plt.savefig(ppj("OUT_FIGURES", "penalty.pdf"))



    # NEW
    # sim_name = sys.argv[1]
    # /home/christopher/Dokumente/rm_fused_lasso/bld/out/analysis
    # with open(ppj("OUT_ANALYSIS", "beta_hat_monte_Carlo_{}.pickle".format(sim_name)), "rb") as in123_file:
    #     aux1234 = pickle.load(in123_file)
    #
    # with open("/home/christopher/Dokumente/rm_fused_lasso/bld/out/analysis/beta_hat_monte_Carlo_two_blocks.pickle", "rb") as in123_file:
    #     aux1234 = pickle.load(in123_file)
    #
    # beta = aux1234[0]
    # beta_container = aux1234[1]
    # s_opt_container = aux1234[2]
    #
    # plt.hist(beta_container[89,:,2], bins='auto')
    # plt.plot(beta)
    # beta[88:100]
    #
    # 393 in 392 out
    # np.shape(beta_container)
    #
    # plt.plot(beta_container[:,9,2])


    # Plot cgh_data
    fig2, ax2 = plt.subplots()
    plt.plot(cgh_data)
    ax2.set_xlim([0, 990])
    ax2.set_ylim([-3, 6])
    plt.xlabel('Genome Order')
    plt.ylabel('Copy Number')
    plt.axhline(color='r')

    plt.plot(cgh_data,zorder = 1)

    plt.savefig(ppj("OUT_FIGURES", "red_line.pdf"))
