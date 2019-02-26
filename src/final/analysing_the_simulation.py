import pickle
import json
import numpy as np
import sys
import matplotlib.pyplot as plt
from bld.project_paths import project_paths_join as ppj


if __name__ == "__main__":
    """ waf """
    sim_name = sys.argv[2]
    reg_name = sys.argv[1]
    sim_dict = json.load(open(ppj("IN_MODEL_SPECS", sim_name + ".json"), encoding="utf-8"))
    with open(ppj("OUT_ANALYSIS", "simulation_{}_{}.pickle".format(reg_name, sim_name)), "rb") as in12_file:
        simulated_data = pickle.load(in12_file)



    # Load Data from pickle file
    beta_hat = simulated_data[0]
    true_beta = simulated_data[1]

    p = sim_dict["p"]
    n = sim_dict["n"]
    s1_min = sim_dict["s1_min"]
    s1_max = sim_dict["s1_max"]
    s2_min = sim_dict["s2_min"]
    s2_max = sim_dict["s2_max"]
    number_blocks = sim_dict['number_of_blocks']
    length_blocks = sim_dict['length_blocks']
    amplitude = sim_dict['amplitude']
    spike_level = sim_dict['spike_level']
    levels = sim_dict['levels']
    spikes = sim_dict['spikes']
    num_simulations = sim_dict['num_simulations']


    container_analysis = []
    #
     # amplitude = 3

    # number of relevant variables
    correct_nonzero = sum((beta_hat >= 0.30*amplitude)  & (true_beta > 0)) * 1 / np.sum(true_beta > 0,axis = 0)
    container_analysis.append(np.mean(correct_nonzero))
    container_analysis.append(np.std(correct_nonzero))



    #number of zeros
    # beta_hat[(np.absolute(beta_hat) <= 0.01) & (true_beta == 0)] = 0
    # number_correct_zero = np.sum((beta_hat <= 0.1*amplitude) & (true_beta == 0),
    #                                                             axis=0)
    #
    # beta_hat[(np.absolute(beta_hat) <= 0.01) & (true_beta == 0)] = 0
    percent_correct_zero = np.sum((np.absolute(beta_hat) <= 0.1) & (true_beta == 0),
                                                                axis=0) / np.sum(true_beta == 0,axis = 0)

    container_analysis.append(np.mean(percent_correct_zero))
    container_analysis.append(np.std(percent_correct_zero))





    #count the spikes
    spike_count = []
    if sim_dict["spikes"] >= 1:
        for i in range(num_simulations): #0 bis 199

            count = 0

            for j in range(p): #0 bis 499

                if j == p-1:
                    break

                if (j == 0) & (true_beta[1,i] == 0) & (beta_hat[j,i] > amplitude/2) & (true_beta[0,i] > 0):

                    count = count + 1

                if (j == (p-1)) & (beta_hat[(p-1), i] > 2) & (true_beta[p-2, i] == 0) & (true_beta[p-1, i] > 0):
                    count = count + 1

                if (true_beta[j-1, i] == 0) & (true_beta[j+1, i] == 0) & (beta_hat[j, i] > amplitude/2) & (true_beta[j, i] > 0):

                    count = count + 1

            spike_count.append(count)
        spike_count = np.array(spike_count) / spikes
        container_analysis.append(np.mean(spike_count))
        container_analysis.append(np.std(spike_count))

    else:
        spike_count = np.zeros(num_simulations)
        container_analysis.append(np.mean(spike_count))
        container_analysis.append(np.std(spike_count))

    # count how many blocks got estimated correctly
    # beta_hat[(beta_hat >= 0.50*amplitude) &
    #          (beta_hat <= 1.5*amplitude) & (true_beta == amplitude)] = amplitude
    # beta_hat[(beta_hat >= 0.75*levels) &
    #          (beta_hat <= 1.25*levels) & (true_beta == levels)] = levels
    # number_of_blocks = (sum((true_beta == beta_hat) & (beta_hat > 0.5))
    #                     - spike_count) / length_blocks
    if reg_name == 'spikes':
        number_blocks = 1


    # if reg_name == 'fusion':
    #     number_of_blocks = np.sum(((beta_hat >= 0.9*amplitude) & (beta_hat <= 1.1x*amplitude) & (true_beta == amplitude)) | ((beta_hat >= 0.9*levels) & (beta_hat <= 1.1*levels) & (true_beta == levels)),axis = 0)
    # else:
    counter_blocks = np.sum(((beta_hat >= 0.50*amplitude) & (beta_hat <= 1.5*amplitude) & (true_beta == amplitude)) | ((beta_hat >= 0.75*levels) & (beta_hat <= 1.25*levels) & (true_beta == levels)),axis = 0)

    percent_blocks = np.array(counter_blocks) / (length_blocks * number_blocks)
    container_analysis.append(np.mean(percent_blocks))
    container_analysis.append(np.std(percent_blocks))

    with open(ppj("OUT_ANALYSIS", "analysis_{}_{}.pickle".format(reg_name, sim_name)), "wb") as out_file:
       pickle.dump(container_analysis, out_file)
