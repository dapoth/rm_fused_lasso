from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt



def obj_unconstrained(b,y,x):
    b = np.array(b)
    
    return np.sum((y-np.dot(x,b)))**2

x = np.random.rand(900,1)
eps = np.random.rand(900,1)
#b = np.random.rand(3,1)


auxaux = np.array([2,9.1])
y = np.dot(np.transpose(x),auxaux)
x0 = np.ones(2)
opt = minimize(obj_unconstrained, x0,args=(y,x), method='Nelder-Mead', options={'xtol': 1e-8, 'disp': True, 'maxiter' : 10000})





aux = np.matmul(np.transpose(x),y)

beta =  np.matmul(np.linalg.inv(np.matmul(np.transpose(x),x)),aux )

plt.scatter(np.array(y),x)