# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 11:45:04 2018

@author: Clara, Cristoph and David
"""
import numpy as np
import random as rm
import scipy as sp
from sklearn import linear_model, model_selection
#import matplotlib as mlp
import matplotlib.pyplot as plt


#Set seeds for replicability. Change seed to get distinct simulations runs.
np.random.seed(1)
rm.seed(7)

### Data generating processs
n = 100
p = 500
positive_betas = 10
amplitude=1    #macht es Sinn unterschiedliche Amplituden einzubauen?
k = 4

p_scale = np.arange(1, p+1)  #zum Plotten als x-Achse für die betas
means = np.zeros(p) #für x, die Regressoren
covs = np.identity(p)
data_indep = np.random.multivariate_normal(means, covs, n)

error = np.random.normal(0,1,n)

beta = np.zeros(p)

index_for_data = rm.sample(range(0,p),positive_betas)
print("Index für relevante Regressoren:",index_for_data)

beta[index_for_data]=amplitude
#print("Oracle Regressoren:",beta)

Y= data_indep.dot(beta) + error



print("Summe der erklärten Variablen:",sum(Y))

#%%

"""
 Crossvalidation using k-fold
 for reference see "https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6"
 
"""


def crossvalidation(alpha):
    
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
        
    return cv_criterion



"""
 Optimization of penalty constant alpha_cv.
 Not clear how to configurate parameters that stay constant and parameters 
 that should be optimised.

"""

alpha_cv = sp.optimize.minimize(crossvalidation, x0 = .2, options={'disp': True})


#%%

"""
 Application of crossvalidated coefficient to our dataset.

"""


reg = linear_model.Lasso(alpha = alpha_cv.x, fit_intercept = False)

reg.fit(data_indep,Y)
reg.predict(data_indep)
print("Geschätzte Koeffizienten:",reg.coef_)

diff= np.subtract(beta,reg.coef_)


#%%
"""
 Plotting our LASSO algorithm against oracle.

"""


plt.figure(1)
plt.subplot(211)
plt.plot( p_scale, reg.coef_, 'r')
plt.title('Lasso Estimation')
plt.ylabel("Coefficients")
plt.subplot(212)

plt.plot( p_scale, beta)
plt.title('Oracle')
plt.ylabel("Coefficients")
plt.xlabel("Features")
plt.show()
print("Schätzfehler:",sp.stats.describe(reg.coef_))
#plt.plot(diff)
print("Varianz:",np.mean(np.square(diff)))




