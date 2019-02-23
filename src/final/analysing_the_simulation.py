import pickle
import json
import numpy as np
import sys
import matplotlib.pyplot as plt
from bld.project_paths import project_paths_join as ppj


if __name__ == "__main__":
    """ waf """
    sim_name = sys.argv[1]
    sim_dict = json.load(open(ppj("IN_MODEL_SPECS", sim_name + ".json"), encoding="utf-8"))
    with open(ppj("OUT_ANALYSIS", "simulated_beta_hat_beta_penalty_cv_y_hat_residuals_{}.pickle".format(sim_name)), "rb") as in12_file:
        simulated_beta_hat_beta_penalty_cv_y_hat_residuals = pickle.load(in12_file)



    # Load Data from pickle file
    beta_hat = simulated_beta_hat_beta_penalty_cv_y_hat_residuals[0]
    beta = simulated_beta_hat_beta_penalty_cv_y_hat_residuals[1]
    penalty_cv =simulated_beta_hat_beta_penalty_cv_y_hat_residuals[2]
    y_hat = simulated_beta_hat_beta_penalty_cv_y_hat_residuals[3]
    residuals = simulated_beta_hat_beta_penalty_cv_y_hat_residuals[4]

    # Load data from json
    amplitude = sim_dict["amplitude"]
    num_simulations = sim_dict["num_simulations"]
    num_of_blocks = sim_dict["num_of_blocks"]
    lenght_of_blocks = sim_dict["lenght_of_blocks"]

    """analysis of estimator properties and how the true beta got estimated"""
    beta_hat[beta_hat <= 0.1*amplitude] = 0
    number_correct_zero = sum((beta_hat == 0) & (beta== 0))

    beta_hat[(beta_hat >= 0.75*amplitude)  & (beta_hat <= 1.25*amplitude)] = amplitude
    number_correct = sum(beta_hat == amplitude)

    #count how many blocks got estimated correctly
    count = 0
    for j in range(num_simulations):
        for i in range(len(beta[:,1])-number_of_blocks):
            if sum(beta[i:i+lenght_of_blocks,j] == amplitude) == lenght_of_blocks:
                count = count +1




    container_ana = [number_correct_zero, number_correct, count]
    with open(ppj("OUT_ANALYSIS", "analysis_{}.pickle".format(sim_name)), "wb") as out_file:
       pickle.dump(container_ana, out_file)
