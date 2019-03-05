"""Load Results from timing simulation and save them as tex table."""
import pickle
import pandas as pd
from bld.project_paths import project_paths_join as ppj

# Load results from timing simulation and add them to RESULTS_TIMING.
RESULTS_TIMING = []
for sim in 'large_blocks', 'blocks_few_spikes', 'small_blocks', 'spikes':
    with open(ppj("OUT_ANALYSIS", "time_list_{}.pickle".format(sim)), "rb") as in_file:
        time_list = pickle.load(in_file)
        time_list.append(sim)
        RESULTS_TIMING.append(time_list)
RESULTS_TIMING = pd.DataFrame(RESULTS_TIMING)
COLUMN_NAMES = {0:"Features", 1:"Number of Obs", 2:"time: CV", 3:"time: Estimation", 4:"Setting"}
RESULTS_TIMING = RESULTS_TIMING.rename(columns=COLUMN_NAMES)
RESULTS_TIMING = RESULTS_TIMING.set_index(["Setting"])

# Transform table to latex and save it.
RESULTS_TIMING.to_latex()
with open(ppj("OUT_FIGURES", "timetable.tex"), 'w') as tf:
    tf.write(RESULTS_TIMING.to_latex())
