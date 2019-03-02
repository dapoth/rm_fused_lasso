#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 13:59:11 2019

@author: clara
"""
import matplotlib.pyplot as plt
import pickle

#   /home/clara/rm_fus…o/bld/out/analysis/beta_hat_large_blocks.pickle
with open("/home/clara/rm_fused_lasso/bld/out/analysis/beta_hat_large_blocks.pickle", "rb") as block1:
    sim_block1 = pickle.load(block1)
beta_hat = sim_block1[0]
beta =sim_block1[1]

with open("/home/clara/rm_fused_lasso/bld/out/analysis/beta_hat_two_blocks.pickle", "rb") as block3:
    sim_block3 = pickle.load(block3)
beta_hat_3 = sim_block3[0]
beta_3 =sim_block3[1]

for i in range(1):
    plt.plot(range(len(beta_hat[:,i])),beta_hat[:,i])
    plt.plot(range(len(beta[:,i])),beta[:,i])
    
plt.show()
for i in range(2,3):
    plt.plot(range(len(beta_hat_3[:,i])),beta_hat_3[:,i])
    plt.plot(range(len(beta_3[:,i])),beta_3[:,i])
plt.show()    
# =============================================================================
# for i in range(20):
#     plt.plot(range(len(beta_hat[:,i])),beta_hat[:,i])
#     plt.plot(range(len(beta[:,i])),beta[:,i])
# =============================================================================

# =============================================================================
# 
#  for i in range(num_simulations):
#         beta_hat[:,i] = fle(penalty_cv[0], penalty_cv[1]).fit(X, y[:,i])
#         y_hat[:,i] = np.matmul(X,beta_hat[:,i])
#         residuals[:,i] = y[:,i] - y_hat[:,i]
# 
#     container = [beta_hat, beta, penalty_cv, y_hat, residuals]
#     with open(ppj("OUT_ANALYSIS", "beta_hat_{}.pickle".format(sim_name)), "wb") as out_file:
#         pickle.dump(container, out_file)
#         
# 
# =============================================================================
