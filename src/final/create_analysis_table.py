"""Load results from simulation analysis and save them as table."""
import pickle
import pandas as pd
from bld.project_paths import project_paths_join as ppj

# Load results from simulation analysis and add them to the list RESULTS.
RESULTS = []
for sim in 'large_blocks', 'blocks_few_spikes', 'small_blocks', 'spikes':
    for reg in 'lasso', 'fused', 'fusion':
        with open(ppj("OUT_ANALYSIS", "analysis_{}_{}.pickle".format(reg, sim)), "rb") as in12_file:
            analysis = pickle.load(in12_file)
            analysis.append(sim)
            analysis.append(reg)
            RESULTS.append(analysis)
RESULTS_PD = pd.DataFrame(RESULTS).round(2)
RESULTS_PD

# Add '(' and ')' around estimated standard errors.
COL_STD_ERR = [1, 3, 5, 7]
for i in COL_STD_ERR:
    for j in range(12):
        RESULTS_PD.iloc[j, i] = '(' + str(RESULTS_PD.iloc[j, i]) + ')'

IDX_NAME = {0:'Lasso', 1:'Fused', 2: 'Fusion', 3:'Lasso', 4:'Fused', 5: 'Fusion',
            6:'Lasso', 7:'Fused', 8: 'Fusion', 9:'Lasso', 10:'Fused', 11: 'Fusion'}
COLUMN_NAME = {0:"sensitivity", 1:"", 2:"specificity", 3:"", 4:"spikes", 5:"",
               6:"blocks", 7:"", 8:"Setting", 9:"Method"}
RESULTS_PD = RESULTS_PD.rename(columns=COLUMN_NAME)




RESULTS_PD = RESULTS_PD.set_index(["Setting", "Method"])


RESULTS_PD.to_latex()

with open(ppj("OUT_FIGURES", "mytable.tex"), 'w') as tf:
    tf.write(RESULTS_PD.to_latex())
