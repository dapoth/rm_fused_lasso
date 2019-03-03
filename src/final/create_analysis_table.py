#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 11:55:21 2019
@author: clara
"""

from tabulate import tabulate
import pickle
import numpy as np
import pandas as pd
from bld.project_paths import project_paths_join as ppj

"""Load analysis data"""


 #data = {'item1': df_oneblock, 'item2': blocks}

list = []
for sim in 'large_blocks', 'blocks_few_spikes', 'small_blocks', 'spikes':
    for reg in 'lasso', 'fused', 'fusion':
          with open(ppj("OUT_ANALYSIS", "analysis_{}_{}.pickle".format(reg, sim)), "rb") as in12_file:
              analysis = pickle.load(in12_file)
              analysis.append(sim)
              analysis.append(reg)
              list.append(analysis)
list_pd = pd.DataFrame(list).round(2)
list_pd

for i in [1,3,5,7]:
    for j in range(12):

        list_pd.iloc[j,i] = '(' + str(list_pd.iloc[j,i]) + ')'

idx_rename = {0:'Lasso', 1:'Fused', 2: 'Fusion',3:'Lasso', 4:'Fused', 5: 'Fusion',6:'Lasso', 7:'Fused', 8: 'Fusion',9:'Lasso', 10:'Fused', 11: 'Fusion'}
columns_rename = {0:"sensitivity" ,1:""  ,2:"specificity", 3:"" , 4:"spikes", 5:"" ,6:"blocks",7:"",8:"Setting",9:"Method"}
list_pd = list_pd.rename( columns=columns_rename)




list_pd = list_pd.set_index(["Setting","Method"])


list_pd.to_latex()

with open(ppj("OUT_FIGURES", "mytable.tex"), 'w') as tf:
     tf.write(list_pd.to_latex())
