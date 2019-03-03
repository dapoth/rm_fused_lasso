import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pandas as pd
import random as rd
from sklearn import model_selection
import math

def crossvalidation(alpha):

    k = 4

    cv_criterion = 0
    kf = model_selection.KFold(n_splits=k)   # Define the split into k-fold
    kf.get_n_splits(data_indep)              # returns the number of splittings (k)
    print(kf)
    for train_index, test_index in kf.split(data_indep):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = data_indep[train_index], data_indep[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        reg = linear_model.Lasso(alpha, fit_intercept= False)

        reg.fit(X_train, Y_train)
        Y_test_hat = reg.predict(X_test)

        cv_criterion += np.sum(np.square(Y_test - Y_test_hat)/len(Y_test))

    cv_criterion
