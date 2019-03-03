import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pandas as pd
import random as rd
from sklearn import model_selection
import math

def solution_path_lasso_signal(y):

    n = len(y)

    lambda_container = np.array([])

    lambda1 = 0
    lambda_container = np.append(lambda_container,[lambda1])

    beta_matrix = np.ones((n,1))
    beta_matrix[:,0] = y



    while sum(y) != 0:

        lambda_container_loop = np.ones(n)

        for i in list(range(n)):

            if y[i] > n*lambda1/2:

                lambda_container_loop[i] = y[i]*2/n



            elif y[i] <-n* lambda1 /2:

                lambda_container_loop[i] = -y[i]*2/n


            else:
                lambda_container_loop[i] = 10000


        lambda_container_loop[lambda_container_loop < lambda1] = 10000
        lambda1 = np.amin(lambda_container_loop)
        lambda_container = np.append(lambda_container,[lambda1])

        y = np.array(return_beta_lasso_signal(y,lambda1))
        y = np.reshape(y, (-1, n)).T
        beta_matrix = np.concatenate((beta_matrix,y),axis = 1)

        for i in list(range(len(lambda_container))):
            plt.plot(lambda_container, beta_matrix[i,:], '.-')




    return beta_matrix, lambda_container
