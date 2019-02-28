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
import cvxpy as cp


if __name__ == "__main__":

    pic_dict = json.load(open(ppj("IN_MODEL_SPECS", "picture_cgh.json"), encoding="utf-8"))
    cgh_data = np.loadtxt(ppj("IN_DATA", "cgh.txt"))
    beta = fused_lasso_primal(cgh_data, np.identity(len(cgh_data)), pic_dict["s1"], pic_dict["s2"])


    plt.xlabel('Genome Order')
    plt.ylabel('Copy Number')
    plt.plot(cgh_data,"bo")
    plt.axhline(color='r')
    plt.savefig(ppj("OUT_FIGURES", "cgh_plot_raw.png"))


    plt.xlabel('Genome Order')
    plt.ylabel('Copy Number')
    plt.plot(cgh_data,"bo")
    plt.axhline(color='r')
    plt.plot(beta, color='orange')
    plt.savefig(ppj("OUT_FIGURES", "cgh_plot_beta.png"))

# plt.show()
#
#
#
#
#
# # Plot cgh_data
#
#
#
#
#
# fig2, ax2 = plt.subplots()
# plt.plot(cgh_data)
# ax2.set_xlim([0, 990])
# ax2.set_ylim([-3, 6])
# plt.xlabel('Genome Order')
# plt.ylabel('Copy Number')
# plt.axhline(color='r')
# plt.plot(beta)
# plt.plot(cgh_data,zorder = 1)
#
# plt.savefig(ppj("OUT_FIGURES", "cgh_plot.png"))
#
# fig, axes = plt.subplots(nrows = 1, ncols = 2)
#
#
# axes[0].set_title('cgh_data')
# axes[0].plot(cgh_data,zorder = 1)
# axes[0].axhline(color='r')
# axes[0].xlabel('Genome Order')
# axes[0].ylabel('Copy Number')
#
# axes[1].plot(cgh_data,zorder = 1)
# axes[1].plot(beta, color='orange')
# axes[1].axhline(color='r')
# axes[1].xlabel('Genome Order')
# axes[1].ylabel('Copy Number')
#
#
#
#
#
#
# cgh_data = np.loadtxt("cgh.txt")
# beta = fused_lasso_primal(cgh_data, np.identity(len(cgh_data)), 160, 15)
# plt.xlabel('Genome Order')
# plt.ylabel('Copy Number')
# plt.plot(cgh_data,"bo")
# plt.axhline(color='r')
# plt.plot(beta, color='orange')
# plt.show()
# plt.savefig("cgh_plot.png")
#
# import matplotlib.pyplot as plt
# import cvxpy as cp
# import numpy as np

# import matplotlib.pyplot as plt
# import cvxpy as cp
# import numpy as np
#
#
#
# def fused_lasso_primal(y,x,s1,s2):
#
#     ### "from constraint import un_constraint" to import function
#     ### y and x data as usual
#
#     p = len(x[1,:])
#     b = cp.Variable(p)
#     error = cp.sum_squares(x*b - y)
#     obj = cp.Minimize(error)
#     constraints = [cp.norm(b,1) <= s1, cp.norm(b[1:p]-b[0:p-1],1) <= s2]
#     prob = cp.Problem(obj, constraints)
#     prob.solve()
#
#     return b.value
#
#
#
# # Plot cgh_data
#
# cgh_data = np.loadtxt("cgh.txt")
# beta = fused_lasso_primal(cgh_data, np.identity(len(cgh_data)), 80, 0.13)
# plt.xlabel('Genome Order')
# plt.ylabel('Copy Number')
# plt.plot(cgh_data,"bo")
# plt.axhline(color='r')
# plt.plot(beta, color='orange')
# plt.show()
# plt.savefig("cgh_plot.png")
