#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 11:55:21 2019
@author: clara
"""

import scipy
from tabulate import tabulate
import pickle
import numpy as np
import pandas as pd
from bld.project_paths import project_paths_join as ppj

"""Load analysis data"""




"""Time Table"""
list_time = []
for sim in 'large_blocks', 'blocks_few_spikes', 'small_blocks', 'spikes':
    with open(ppj("OUT_ANALYSIS", "time_list_{}.pickle".format(sim)), "rb") as in12_file:
        time_list = pickle.load(in12_file)
        time_list.append(sim)
        list_time.append(time_list)
time_pd = pd.DataFrame(list_time)
columns_rename = {0:"p"   ,1:"n", 2:"time: cv" , 3:"time: estimation",4:"Setting"}
time_pd = time_pd.rename( columns=columns_rename)
time_pd = time_pd.set_index(["Setting"])

time_pd.to_latex()

with open(ppj("OUT_FIGURES", "timetable.tex"), 'w') as tf:
     tf.write(time_pd.to_latex())
