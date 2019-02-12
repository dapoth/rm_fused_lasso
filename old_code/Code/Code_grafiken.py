import matplotlib.pyplot as plt
import numpy as np
from easy_lasso import solution_path
import random as rd
import pandas as pd

data = np.loadtxt('cgh.txt')

fig2, ax2 = plt.subplots()
plt.plot(data)
ax2.set_xlim([0, 990])
ax2.set_ylim([-3, 6])
plt.xlabel('Genome Order')
plt.ylabel('Copy Number')
plt.axhline(color='r')



#plt.setp(lines,color='r',linewidth=4.0)
#plt.plot([-1,0,1],[0,0,0],color='r',linewidth=4.0,zorder = 2)
plt.plot(data,zorder = 1)

#plt.savefig('D:/Research_Module/Vortrag/red_line.pdf')

#plt.savefig('D:\Research_Module\Vortrag\cgh_motivation_lasso.pdf')



#absolute value
plt.xlabel('x')
plt.ylabel('|x|')
plt.plot([-1,0,1], [1,0,1])
#plt.savefig('D:/Research_Module/Vortrag/absolute.pdf')




fused = np.loadtxt('cgh_container.txt')
lambda2 = np.loadtxt('lambda_vector.txt') 
plt.plot(data)
plt.plot(fused[:,915])
plt.plot(lambda2[860])
#plt.savefig('D:/Research_Module/Vortrag/cgh_fused.pdf')



#lasso easy data
beta_1,lambda_11 =solution_path(data)

fig1, ax1 = plt.subplots()
plt.plot(beta_1[:,55]) #50
ax1.set_xlim([0, 990])
ax1.set_ylim([-3, 6])
plt.xlabel('Beta')
plt.ylabel('Value')

#plt.savefig('D:/Research_Module/Vortrag/cgh_lasso.pdf')

#lasso easy y
np.random.seed(40)
rd.seed(5)
y = np.random.normal(0, 20, 30)
beta2, lambda2 = solution_path(y)
plt.plot(beta2[:,6],lambda2)

beta_matrix123 = pd.DataFrame(beta2)
lambda_container = pd.DataFrame(lambda2)
for i in list(range(len(lambda_container))):
    
    
    
    d = beta_matrix123.iloc[i,0:len(lambda_container)]
    c = lambda_container
    #s = UnivariateSpline(c, d,k=1)
    plt.xlabel('Lambda')
    plt.ylabel('Value of beta')
    plt.plot(c, d, '.-')
    plt.axvline(x=0.27,color='black',linestyle='dotted')
    #plt.savefig('D:/Research_Module/Vortrag/lasso_solution_path2pdf')
    



