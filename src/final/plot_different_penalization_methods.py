import matplotlib.pyplot as plt
import pickle
import numpy as np
import math
import random as rd
import matplotlib.pyplot as plt
from src.model_code.functions import fused_lasso_dual
from bld.project_paths import project_paths_join as ppj



list_true_beta = []
list_beta_hat = []

for reg in 'lasso', 'fused', 'fusion':
    for sim in 'blocks_levels', 'blocks_few_spikes', 'blocks_many_spikes', 'spikes':

        with open(ppj("OUT_ANALYSIS", "simulation_{}_{}.pickle".format(reg, sim)), "rb") as in12_file:
             simulation_data = pickle.load(in12_file)
        # with open("/home/christopher/Dokumente/rm_fused_lasso/bld/out/analysis/simulation_{}_{}.pickle".format(reg,sim), "rb") as in12_file:
        #     simulation_data = pickle.load(in12_file)

        beta_hat = simulation_data[0][:, 1]
        true_beta = simulation_data[1][:, 1]

        list_true_beta.append(true_beta)
        list_beta_hat.append(beta_hat)

    fig, axes = plt.subplots(2, 2, gridspec_kw = {"height_ratios": [1,1]})


    axes[0,0].set_title('blocks_levels')
    axes[0, 0].plot(list_true_beta[0])
    axes[0, 0].plot(list_beta_hat[0])

    axes[1,0].set_xlabel('blocks_few_spikes')
    axes[1, 0].plot(list_true_beta[1])
    axes[1, 0].plot(list_beta_hat[1])

    axes[0, 1].plot(list_true_beta[2])
    axes[0, 1].plot(list_beta_hat[2])
    axes[0,1].set_title('blocks_many_spikes')

    axes[1, 1].plot(list_true_beta[3])
    axes[1, 1].plot(list_beta_hat[3])
    axes[1,1].set_xlabel('Spikes')

    list_true_beta = []
    list_beta_hat = []




    plt.savefig(ppj("OUT_FIGURES", "plot_{}.pdf".format(reg)))













p = 300
n = 100
betas = np.zeros(300)

betas[10:20] = 2
betas[50:60] = 1
betas[200:230] = 1.5
betas[240:260] = 1
betas[280:290] = 1

mean = np.zeros(p)
cov = np.identity(p)
X = np.random.multivariate_normal(mean, cov, n)
eps = np.random.randn(n)
Y = np.matmul(X, betas) + eps

beta_lasso = fused_lasso_dual(Y,X,500,0)

beta_fusion = fused_lasso_dual(Y,X,0,50)

beta_fused = fused_lasso_dual(Y,X,100,300)

beta_both = fused_lasso_dual(Y,X,250,250)


fig, axes = plt.subplots(2, 2, gridspec_kw = {"height_ratios": [1,1]})


axes[0,0].set_title('Lasso')
axes[0, 0].plot(betas)
axes[0, 0].plot(beta_lasso)

axes[1,0].set_xlabel('Fusion')
axes[1, 0].plot(betas)
axes[1, 0].plot(beta_fusion)

axes[0, 1].plot(betas)
axes[0, 1].plot(beta_fused)
axes[0,1].set_title('Fused')

plt.savefig(ppj("OUT_FIGURES", "different_penalization_methods.pdf"))
