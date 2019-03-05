"""Calculate fused lasso estimates for CGH data and plot them."""
import json
import matplotlib.pyplot as plt
import numpy as np
from bld.project_paths import project_paths_join as ppj
from src.model_code.fused_lasso_primal import fused_lasso_primal


if __name__ == "__main__":

    PIC_DICT = json.load(open(ppj("IN_MODEL_SPECS", "picture_cgh.json"), encoding="utf-8"))
    CGH_DATA = np.loadtxt(ppj("IN_DATA", "cgh.txt"))
    BETA = fused_lasso_primal(CGH_DATA, np.identity(len(CGH_DATA)), PIC_DICT["s1"], PIC_DICT["s2"])


    plt.xlabel('Genome Order')
    plt.ylabel('Copy Number')
    plt.plot(CGH_DATA, "bo")
    plt.axhline(color='r')
    plt.savefig(ppj("OUT_FIGURES", "cgh_plot_raw.png"))

    plt.xlabel('Genome Order')
    plt.ylabel('Copy Number')
    plt.plot(CGH_DATA, "bo")
    plt.axhline(color='r')
    plt.plot(BETA, color='orange')
    plt.savefig(ppj("OUT_FIGURES", "cgh_plot_beta.png"))
