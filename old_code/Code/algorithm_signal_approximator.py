import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
import time
from numba import jit
from functions import return_beta_unconstraint

start_time = time.time()


@jit(nopython=True)
def group_minimize_np_test(a,b,y,lambda2,p):

    container12 = np.ones((len(b),2))

    container12[:,0] = a
    container12[:,1] = b

    beta123 = np.ones(p)

    # Init
    number_of_groups2 = len(container12)

    for z in list(range(len(b))):

        stretch = np.ones(np.int(container12[z,1])) * container12[z,0]
        auxauxaux = np.cumsum(container12[:,1])

        if z == 0:
            beta123[0:int(container12[0,1])] = stretch

        else:
            beta123[int(auxauxaux[z-1]):int(auxauxaux[z])] = stretch


    return 0.5*(np.sum((y-beta123)**2))+lambda2 * np.sum(np.abs(container12[:,0][0:number_of_groups2-1]-container12[:,0][1:number_of_groups2]))

def group_minimize_np_true(a,b,y,lambda2,p):
    
    container12 = np.ones((len(b),2))
    
    container12[:,0] = a
    container12[:,1] = b
    
   # beta123 = np.ones(p)
    
    number_of_groups2 = len(container12)
    
    
    beta123 = np.repeat(a,b.astype(int))
    
   
            
            
    return 0.5*(np.sum((y-beta123)**2))+lambda2 * np.sum(np.abs(container12[:,0][0:number_of_groups2-1]-container12[:,0][1:number_of_groups2]))
        



def fused_signal_approximation(y):
   
    
  
    # Initialization
    p = len(y)
    beta = np.array(y)
    
    container = np.ones((p,2))
    container[:,0] = y
    container[:,1] = np.ones(p)
    
    beta_matrix = np.ones((p,1))
    beta_matrix[:,0] = y
    
    
    lambda_vector = np.array([])
    
    lambda2 = 0
    lambda_vector = np.append(lambda_vector,[lambda2])
    
    kk = 0
    
    while lambda2 < 5000:#for i in list(range(40)):#len(container[:,0]) > 1:
        
        kk = kk +1
        print(kk)
        
        group_count = len(container[:,1])
        
        ###Calculating hitting time
        
        ## Calculating differences (zähler)
        difference = beta[0:group_count-1] - beta[1:group_count]
        
        
        ##calculate derivatives
        beta_deriv_container = np.ones(group_count)
        
        for j in list(range(group_count)):
            
            
            
            if j == 0:
                beta_deriv_container[j] = (-1/container[j,1]) * (np.sign(container[j,0]-np.sign(container[j+1,0])))
            elif j == len(container[:,1])-1:
                beta_deriv_container[j] = (-1/container[j,1]) * (np.sign(container[j,0]-np.sign(container[j-1,0])))
            else:
                beta_deriv_container[j] = (-1/container[j,1])* (np.sign(container[j,0]-np.sign(container[j-1,0])+ np.sign(container[j,0]-np.sign(container[j+1,0]))))
         
            
        ## Calculating differences in derivatives    (Nenner)
        difference_clean_deriv = beta_deriv_container[1:group_count] - beta_deriv_container[0:group_count-1]
        
        ##calculating hitting time
        h = np.divide(difference,difference_clean_deriv) + lambda2
        
        
        # min nur über über Werte größer als das alte lambda sind
        h[h < lambda2] = 10000
        
        lambda2 = np.amin(h)
        
        
        
        #sammeln der lambdas in jeder iteration
        lambda_vector = np.append(lambda_vector,[lambda2])
        
        
        
        ## hitting time indices
        fuse_set_1 = np.argmin(h)
        fuse_set_2 = np.argmin(h) + 1
        
        
        ## update container group column
        container[fuse_set_1,1] = container[fuse_set_1,1]   + container[fuse_set_2,1]
        
        container = np.delete(container, (fuse_set_2), axis=0)
        #print(container)
        
        # startwert für den optimierer
        
        #x0 = np.ones(len(container[:,1]))
        x0 = container[:,0]
        
        ##update beta
        beta_opt = minimize(group_minimize_np_true, x0,args=(container[:,1],y,lambda2,p), method='Nelder-Mead', options={'xtol': 1e-8, 'disp': True, 'maxiter' : 100000})
        
        
        
       
        ## add new beta to container
        container[:,0] = beta_opt['x']
        
        
        ### For given updated container fill beta_matrix
        
        aux = np.array([np.repeat(container[:,0],container[:,1].astype(int))]).T
        
        beta_matrix = np.concatenate((beta_matrix,aux),axis =1)
        
        

    
    # PLot 
    beta_matrix = pd.DataFrame(beta_matrix)
    lambda_vector = pd.DataFrame(lambda_vector)
    y = pd.DataFrame(y)
    for i in list(range(len(lambda_vector))):
        
        
        
        d = beta_matrix.iloc[i,0:len(lambda_vector)-1]
        c = lambda_vector.iloc[0:-1]
        s = UnivariateSpline(c, d,k=1)
        
        plt.plot(c, d, '.-')
        
    return plt.plot(c, d, '.-'), beta_matrix,lambda_vector, container
        
y = np.random.randn(10)      
Plot, beta_Matrix,lambda_vector, container = fused_signal_approximation(y)

#fused_signal_approximation(data[0:100])


#print(Plot)      
 
#print(lambda_vector)

#print(pd.DataFrame(beta_Matrix))

#print(container)   
                                                            
#print("My program took", time.time() - start_time, "to run"  )  


#plt.plot(y)
#plt.plot(beta_Matrix.iloc[:,10])    
       