import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pandas as pd
import random as rd
from sklearn import model_selection 


def solution_path_constraint(y,x,lambda1=0,lambda2=0):
    
    ### "from constraint import constraint" to import function
    ### y and x data as usual
    ### lambda1 and lambda2 optional to make vertical line in the plot
    
    p = len(x[1,:])
    gamma1 = cp.Parameter(nonneg=True)
    gamma2 = cp.Parameter(nonneg=True)
    b = cp.Variable(p)
    error = cp.sum_squares(x*b - y)
    obj = cp.Minimize(error+gamma1*cp.norm(b,1)  +gamma2*cp.norm(b[1:p]-b[0:p-1],1))
    prob = cp.Problem(obj)
    
    
    x_values = []
    gamma_vals = np.logspace(-2, 6)
    for val in gamma_vals:
        gamma1.value = val
        gamma2.value = lambda2
        prob.solve()
        x_values.append(b.value)
    
    x2_values = []    
    gamma2_vals = np.logspace(-2,6)
    for val in gamma_vals:
        gamma1.value = lambda1
        gamma2.value = val
        prob.solve()
        x2_values.append(b.value)
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=(6,10))
    
    # Plot entries of x vs. lambda1.
    plt.subplot(211)
    plt.axvline(x=lambda1)
    for i in range(p):
        plt.plot(gamma_vals, [xi[i] for xi in x_values])
    plt.xlabel(r'$\lambda_1$', fontsize=16)
    plt.ylabel(r'$\beta_{i}$', fontsize=16)
    plt.xscale('log')
    plt.title(r'Entries of $\beta$ vs. $\lambda_1$')
    
    plt.subplot(212)
    plt.axvline(x=lambda2)
    for i in range(p):
        plt.plot(gamma2_vals, [xi[i] for xi in x2_values])
    plt.xlabel(r'$\lambda_2$', fontsize=16)
    plt.ylabel(r'$\beta_{i}$', fontsize=16)
    plt.xscale('log')
    plt.title(r'Entries of $\beta$ vs. $\lambda_2$')
    
    plt.tight_layout()
    plt.show()
    
    
    
    
    #prob.solve()
    
    return print("The prcoess was",prob.status)


def un_constraint(y,x,s1,s2):
    
    ### "from constraint import un_constraint" to import function
    ### y and x data as usual
    
    p = len(x[1,:])
    b = cp.Variable(p)
    error = cp.sum_squares(x*b - y)
    obj = cp.Minimize(error)
    constraints = [cp.norm(b,1) <= s1, cp.norm(b[1:p]-b[0:p-1],1) <= s2]
    prob = cp.Problem(obj, constraints)
    prob.solve()
    
    return prob.status, prob.value, b.value

def return_beta_unconstraint(y,x,lambda1,lambda2):
    
    p = len(x[1,:])
    b = cp.Variable(p)
    error = cp.sum_squares(x*b - y)
    obj = cp.Minimize(error+lambda1*cp.norm(b,1)  +lambda2*cp.norm(b[1:p]-b[0:p-1],1))
    prob = cp.Problem(obj)
    prob.solve()
    
    return b.value

def return_beta_constraint(y,x,s1,s2):
    
    ### "from constraint import un_constraint" to import function
    ### y and x data as usual
    
    p = len(x[1,:])
    b = cp.Variable(p)
    error = cp.sum_squares(x*b - y)
    obj = cp.Minimize(error)
    constraints = [cp.norm(b,1) <= s1, cp.norm(b[1:p]-b[0:p-1],1) <= s2]
    prob = cp.Problem(obj, constraints)
    prob.solve()
    
    return b.value

def return_beta_lasso_signal(y,lambdaa):
    y = np.array(y)
    n = len(y)
    
    y[y>n*lambdaa/2] = y[y>n*lambdaa/2] - n*lambdaa/2
    y[y<-n*lambdaa/2] = y[y<-n*lambdaa/2] + n*lambdaa/2
    y[np.abs(y) <= n*lambdaa/2] = 0
    
    b = y
    
    return b

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





