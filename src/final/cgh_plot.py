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
from src.model_code.functions import fused_lasso_primal
import json



if __name__ == "__main__":

    pic_dict = json.load(open(ppj("IN_MODEL_SPECS", "picture_cgh.json"), encoding="utf-8"))




    # Plot cgh_data
    cgh_data = np.loadtxt(ppj("IN_DATA", "cgh.txt"))
    beta = fused_lasso_primal(cgh_data, np.identity(len(cgh_data)), pic_dict["s1"], pic_dict["s2"])
    fig2, ax2 = plt.subplots()
    plt.plot(cgh_data)
    ax2.set_xlim([0, 990])
    ax2.set_ylim([-3, 6])
    plt.xlabel('Genome Order')
    plt.ylabel('Copy Number')
    plt.axhline(color='r')
    plt.plot(beta)
    plt.plot(cgh_data,zorder = 1)

    plt.savefig(ppj("OUT_FIGURES", "cgh_plot.pdf"))
