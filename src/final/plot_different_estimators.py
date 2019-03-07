"""Plot beta estimations for all settings and estimators.

For each combination of the four settings and three estimators create plots of
the estimated betas for one simulation step. Arrange the plots in a two by two
manner for each setting.

"""
import pickle
import matplotlib.pyplot as plt
from bld.project_paths import project_paths_join as ppj

for reg in 'lasso', 'fused', 'fusion':
    list_true_beta = []
    list_beta_hat = []
    for sim in 'large_blocks', 'blocks_few_spikes', 'small_blocks', 'spikes':

        with open(ppj("OUT_ANALYSIS", "simulation_{}_{}.pickle".
                      format(reg, sim)), "rb") as in_file:
            simulation_data = pickle.load(in_file)

        beta_hat = simulation_data[0][:, 1]
        true_beta = simulation_data[1][:, 1]

        list_true_beta.append(true_beta)
        list_beta_hat.append(beta_hat)

    fig, axes = plt.subplots(2, 2, gridspec_kw={"height_ratios": [1, 1]})

    axes[0, 0].set_title('large_blocks')
    axes[0, 0].plot(list_true_beta[0])
    axes[0, 0].plot(list_beta_hat[0])

    axes[1, 0].set_xlabel('blocks_few_spikes')
    axes[1, 0].plot(list_true_beta[1])
    axes[1, 0].plot(list_beta_hat[1])

    axes[0, 1].plot(list_true_beta[2])
    axes[0, 1].plot(list_beta_hat[2])
    axes[0, 1].set_title('small_blocks')

    axes[1, 1].plot(list_true_beta[3])
    axes[1, 1].plot(list_beta_hat[3])
    axes[1, 1].set_xlabel('Spikes')

    plt.savefig(ppj("OUT_FIGURES", "plot_{}.png".format(reg)))
    plt.clf()
