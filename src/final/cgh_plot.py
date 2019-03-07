"""Calculate fused lasso estimates for CGH data and plot them."""
import matplotlib.pyplot as plt
import numpy as np
from bld.project_paths import project_paths_join as ppj
from src.model_code.fused_lasso_primal import fused_lasso_primal


if __name__ == "__main__":

    #Set penalty constants and CGH data and estimate fused lasso results.
    CGH_DATA = np.loadtxt(ppj("IN_DATA", "cgh.txt"))
    BETA_HAT = fused_lasso_primal(CGH_DATA, np.identity(len(CGH_DATA)),
                                  160, 15)

    # Plot CGH data.
    plt.xlabel('Genome Order')
    plt.ylabel('Copy Number')
    plt.plot(CGH_DATA, "bo")
    plt.axhline(color='r')
    plt.savefig(ppj("OUT_FIGURES", "cgh_plot_raw.png"))

    # Plot CGH data with fused lasso estimates.
    plt.xlabel('Genome Order')
    plt.ylabel('Copy Number')
    plt.plot(CGH_DATA, "bo")
    plt.axhline(color='r')
    plt.plot(BETA_HAT, color='orange')
    plt.savefig(ppj("OUT_FIGURES", "cgh_plot_beta.png"))
