import sys
import json
import logging
import pickle
import numpy as np
import cvxpy as cp
from bld.project_paths import project_paths_join as ppj
import matplotlib.pyplot as plt
from src.model_code.functions import fused_lasso_primal
from src.model_code.functions import solution_path_unconstraint



if __name__ == "__main__":
    model_name = sys.argv[1]
    model = json.load(open(ppj("IN_MODEL_SPECS", model_name + ".json"), encoding="utf-8"))

    logging.basicConfig(
        filename=ppj("OUT_ANALYSIS", "log", "simulation_{}.log".format(model_name)),
        filemode="w",
        level=logging.INFO
    )
    #np.random.seed(1) #model["rng_seed"]
    logging.info(model['penalty1']) #"rng_seed"


    lambda1 = model['penalty1']
    lambda2 = model['penalty2']
    y = np.loadtxt(ppj("OUT_DATA", "y.csv"), delimiter=",")
    X = np.array(np.loadtxt(ppj("OUT_DATA", "x.csv"), delimiter=","))
    X = np.reshape(X,[100,10])
    beta = np.loadtxt(ppj("OUT_DATA", "beta.csv"), delimiter=",")
    eps = np.loadtxt(ppj("OUT_DATA", "eps.csv"), delimiter=",")
    aux = fused_lasso_primal(y,X,lambda1,lambda2)
    with open(ppj("OUT_ANALYSIS", "sim123_{}.pickle".format(model_name)), "wb") as out123_file:
        pickle.dump(aux, out123_file)


        rng = np.random.RandomState(10)
        np.shape(np.hstack((rng.normal(size=1000), rng.normal(loc=5, scale=2, size=1000))))
