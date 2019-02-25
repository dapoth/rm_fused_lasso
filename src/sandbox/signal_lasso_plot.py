#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 15:40:44 2019

@author: clara
"""

import numpy as np
import matplotlib.pyplot as plt
from bld.project_paths import project_paths_join as ppj

anzahl = 100
x_negative=np.linspace(-3, -1.5, 30)
x_zero = np.linspace(-1.5,1.5,60)
x_positive = np.linspace(1.5,3,30)

y_negative = x_negative + 1.5
y_zero = x_zero * 0
y_positive = x_positive - 1.5
x_ols = np.linspace(-3,3,120)

plt.figure() 
plt.plot(x_negative,y_negative, 'b')
plt.plot(x_zero, y_zero, 'b')
plt.plot(x_positive, y_positive, 'b')
plt.plot(x_ols, x_ols, 'k--')
plt.xlabel('beta_OLS')
plt.ylabel('beta_Lasso')
plt.grid(True)
plt.savefig(ppj("OUT_FIGURES", "signal_lasso_plot.pdf"))
plt.show()

 