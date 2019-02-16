import sys
import json
import logging
import pickle
import numpy as np
import cvxpy as cp
from src.model_code.functions import fused_lasso_primal
from bld.project_paths import project_paths_join as ppj


if __name__ == "__main__":
    sim_name = sys.argv[1]
    model = json.load(open(ppj("IN_MODEL_SPECS", sim_name + ".json"), encoding="utf-8"))
    sim_dict = json.load(open(ppj("IN_MODEL_SPECS", sim_name + ".json"), encoding="utf-8"))
    with open(ppj("OUT_ANALYSIS", "simulation_{}.pickle".format(sim_name)), "rb") as in_file:
        data = pickle.load(in_file)

    beta = data[0]
    beta_hat = data[1]
    X = data[2]
    X_T = data[3]
    epsilion = data[4]
    y = data[5]
    num_simulations = sim_dict['num_simulations']
    s_1 = sim_dict['s1']
    s_2 = sim_dict['s2']

    for sim in range(num_simulations):
        beta_hat[:, sim] = fused_lasso_primal(y[:, sim], X, s_1, s_2)

    y_hat = np.matmul(X, beta_hat)
    epsilon_hat = y - y_hat
