import matplotlib.pyplot as plt
import numpy as np
from bld.project_paths import project_paths_join as ppj

if __name__ == "__main__":
    cgh_data = np.loadtxt(ppj("IN_DATA", "cgh.txt"))  

    fig2, ax2 = plt.subplots()
    plt.plot(cgh_data)
    ax2.set_xlim([0, 990])
    ax2.set_ylim([-3, 6])
    plt.xlabel('Genome Order')
    plt.ylabel('Copy Number')
    plt.axhline(color='r')

    plt.plot(cgh_data,zorder = 1)

    plt.savefig(ppj("OUT_FIGURES", "red_line.pdf"))
    
