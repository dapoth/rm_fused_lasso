import seaborn
import sys
import json
import logging
import pickle
import numpy as np
import cvxpy as cp
from src.model_code.flestimator import FusedLassoEstimator as fle
from src.model_code.functions import fused_lasso_primal
from src.model_code.functions import generate_blocks
from src.model_code.functions import generate_data
from bld.project_paths import project_paths_join as ppj
from sklearn.model_selection import GridSearchCV
import random as  rd
import math

def generate_blocks( p, number_blocks, length_blocks, amplitude,
                    spike_level, levels = False, spikes = 0):
    """Generate beta's for simulation purpose."""
    container = np.zeros( p)
    max_blocks = math.floor( p / length_blocks)

    start_blocks = rd.sample( range( max_blocks), number_blocks)

    amplitudes = [amplitude, amplitude*2]

    if (levels == True):
        """
        If the Blocks should not all have equal levels, we will randomly chose
        the level of each block as either amplitude or amplitude times 2.
        """

        for block in start_blocks :
            amp = rd.choice(amplitudes)
            for i in range(p) :
                if (i > (block-1) * length_blocks) and (i <= block * length_blocks):
                    container[i] = amp

    else:
        for block in start_blocks :
            #amp = rd.choice(amplitudes)
            for i in range(p) :
                if ( i > ( block-1)* length_blocks) and (i <= block* length_blocks):
                    container [i] = amplitude

    if spikes != 0 :
        non_blocks = []
        for i in range(p):
            if container[i]==0:
                non_blocks.append(i)
        beta_spikes = rd.sample(non_blocks, spikes)
        for i in beta_spikes:
            container[i] = spike_level


    return container

from sklearn.base import BaseEstimator, RegressorMixin
import cvxpy as cp
import numpy as np
import math
import random as rd
from sklearn.model_selection import GridSearchCV
import inspect
import matplotlib.pyplot as plt

class FusedLassoEstimator(BaseEstimator, RegressorMixin):
    """
    Fused Lasso Estimator that makes a penalized least squares regression
    where diffferences between neighboring betas and non-zero betas get
    penalized.
    The optimization objective for Fused Lasso is:
        (1 / (2 * n_samples)) * ||y - Xb||^2_2
        s.t.        ||b||_1     <= s1
             sum(|b_i-b_{i-1]|) <= s2
    Technically this is a Convex optimization problem that is solved by cvxpy.
    """

    def __init__( self, s1, s2):
        """ Called when initializing the Fused Lasso Estimator. """
        self.s1 = s1
        self.s2 = s2


    def fit( self, X, y = None):
        """
        The code for the fused lasso estimator. The penalties s1 and s2 are
        included as additional restriction equations.
        Examples
    --------
    >>> reg = FusedLassoEstimator( s1 = 1, s2 = 10).fit(X, y)

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """

#        if X.shape[0] != y.shape[0]:
#            raise ValueError( "X and y have inconsistent dimensions (%d != %d)"
#                             % (X.shape[0], y.shape[0]))


        p = len(X[1,:])
        b = cp.Variable(p)
        error = cp.sum_squares(X*b - y)
        obj = cp.Minimize(error)
        constraints = [cp.norm(b,1) <= self.s1, cp.norm(b[1:p]-b[0:p-1],1) <= self.s2]
        prob = cp.Problem(obj, constraints)
        prob.solve()


        self.beta = b.value

        return b.value
    def predict( self, X, beta = None):
        #try:
        #    getattr(self, "treshold_")
        #except AttributeError:
        #    raise RuntimeError("You must train classifer before predicting data!")


        return np.matmul(X, self.beta)

def fused_lasso_primal(y,x,s1,s2):

    ### "from constraint import un_constraint" to import function
    ### y and x data as usual

    p = len(x[1,:])
    b = cp.Variable(p)
    error = cp.sum_squares(x*b - y)
    obj = cp.Minimize(error)
    constraints = [cp.norm(b,1) <= s1, cp.norm(b[1:p]-b[0:p-1],1) <= s2]
    prob = cp.Problem(obj, constraints)
    prob.solve()

    return b.value

if __name__ == "__main__":

""" waf """
# mont_name = sys.argv[0]
# sim_dict = json.load(open(ppj("IN_MODEL_SPECS", "blocks_levels_monte_carlo.json"), encoding="utf-8"))





#### blocks_levels
blocks_levels ={
    "n": [10,100, 1000],
    "p": 200,
    "number_of_blocks":1,
    "amplitude":3,
    "spikes": 0,
    "length_blocks":50,
    "spike_level":0,
    "levels":1,
    "s1":50,
    "s2":50,
    "s1_min":100,
    "s1_max":200,
    "s2_min":12,
    "s2_max":90,
    "num_simulations": 200,
    "num_simulations_mont": 10
}

num_simulations = blocks_levels['num_simulations']
p = blocks_levels['p']
n = blocks_levels['n']
number_blocks = blocks_levels['number_of_blocks']
length_blocks = blocks_levels['length_blocks']
amplitude = blocks_levels['amplitude']
spike_level = blocks_levels['spike_level']
levels = blocks_levels['levels']
spikes = blocks_levels['spikes']

beta = generate_blocks(p, number_blocks, length_blocks, amplitude,
                    spike_level,levels, spikes)


beta_container = np.ones((blocks_levels["p"], blocks_levels["num_simulations"], len(blocks_levels["n"])))
s_opt_container = np.zeros([2,len(blocks_levels["n"])])

for k in list(range(len(blocks_levels["n"]))):

    mean_x = np.zeros(blocks_levels["p"])
    cov_X = np.identity(blocks_levels["p"])
    X = np.random.multivariate_normal(mean_x, cov_X,blocks_levels["n"][k])

    eps = np.random.randn(blocks_levels["n"][k])

    y = X.dot(beta)+eps

    lasso_grid = {
      's1': list(np.linspace(blocks_levels['s1_min'],blocks_levels['s1_max'],20))
    }
    fused_grid = {
      's2': list(np.linspace(blocks_levels['s2_min'],blocks_levels['s2_max'],20))
    }

    two_d_grid = [{
                's1': list(np.linspace(blocks_levels['s1_min'],blocks_levels['s1_max'],20)),
                's2': list(np.linspace(blocks_levels['s2_min'],blocks_levels['s2_max'],20))
                }]

    clf = GridSearchCV(FusedLassoEstimator(lasso_grid,fused_grid), two_d_grid,
                            scoring= 'neg_mean_squared_error',
                            n_jobs = -1, iid= False, refit=True,
                            cv=None, verbose=0, pre_dispatch='2*n_jobs',
                            error_score='raise-deprecating',
                            return_train_score='warn')

    clf.fit(X, y)


    s1 = clf.best_params_ ["s1"]
    s2 = clf.best_params_["s2"]

    s_opt_container[0,k] = s1
    s_opt_container[1,k] = s2
    for index ,i in enumerate(blocks_levels["n"]):  # i =10 i = 1000 usw

        mean_x = np.zeros(blocks_levels["p"])
        cov_X = np.identity(blocks_levels["p"])
        X = np.random.multivariate_normal(mean_x, cov_X,blocks_levels["n"][k])

        for j in range(blocks_levels["num_simulations"]):

            eps = np.random.rand(blocks_levels["n"][k])

            y = np.matmul(X,beta)+eps

            beta_container[:,j,index] = fused_lasso_primal(y,X,s1,s2)

test1234 = [beta, beta_container, s_opt_container]

with open("/home/christopher/Dokumente/Testbereich/cv_results_params.pickle", "wb") as out_file:
    pickle.dump(clf.cv_results_['params'], out_file)

    with open("/home/christopher/Dokumente/Testbereich/cv_results_mean_test_score.pickle", "wb") as out_file:
        pickle.dump(clf.cv_results_['mean_test_score'], out_file)

with open("/home/christopher/Dokumente/Testbereich/cv_results.pickle", "rb") as out_file:
    cv_results = pickle.load(out_file)

clf.cv_results_['mean_test_score']
clf.cv_results_['params']

clf.best_params_

plt.plot(beta)
plt.plot(beta_container[:,1,2])
beta[45:55]
fig, axes = plt.subplots(2, 2)

axes[0,0].set_title('center')
axes[0, 0].hist(beta_container[75,:,0])

axes[1,0].set_xlabel('zero')
axes[1, 0].hist(beta_container[150,:,0])

axes[0, 1].hist(beta_container[51,:,0])
axes[0,1].set_title('block_in')

axes[1, 1].hist(beta_container[50,:,0])
axes[1,1].set_xlabel('block_out')

for i in range(3):
    plt.plot(beta_container[:,:,i])

with open(ppj("OUT_ANALYSIS", "beta_hat_monte_Carlo_{}.pickle".format(mont_name)), "wb") as out_file:
    pickle.dump(test1234, out_file)



        #block_spikes
    block_levels ={
        "n": [10, 100],
        "p": 200,
        "number_of_blocks": 3,
        "amplitude": 3,
        "length_blocks": 10,
        "spike_level": 0,
        "spikes": 0,
        "levels": 1,
        "s1_min": 40,
        "s1_max": 200,
        "s2_min": 3,
        "s2_max": 100,
        "num_simulations": 100
    }

num_simulations = block_levels['num_simulations']
p = block_levels['p']
n = block_levels['n']
number_blocks = block_levels['number_of_blocks']
length_blocks = block_levels['length_blocks']
amplitude = block_levels['amplitude']
spike_level = block_levels['spike_level']
levels = block_levels['levels']
spikes = block_levels['spikes']

beta = generate_blocks(p, number_blocks, length_blocks, amplitude,
                    spike_level,levels, spikes)

plt.plot(beta)
beta_container = np.ones((block_levels["p"], block_levels["num_simulations"], len(block_levels["n"])))
s_opt_container = np.zeros([2,len(block_levels["n"])])

for k in list(range(len(block_levels["n"]))):

    mean_x = np.zeros(block_levels["p"])
    cov_X = np.identity(block_levels["p"])
    X = np.random.multivariate_normal(mean_x, cov_X,block_levels["n"][k])

    eps = np.random.randn(block_levels["n"][k])

    y = X.dot(beta)+eps

    lasso_grid = {
      's1': list(np.linspace(block_levels['s1_min'],block_levels['s1_max'],20))
    }
    fused_grid = {
      's2': list(np.linspace(block_levels['s2_min'],block_levels['s2_max'],20))
    }

    two_d_grid = [{
                's1': list(np.linspace(block_levels['s1_min'],block_levels['s1_max'],20)),
                's2': list(np.linspace(block_levels['s2_min'],block_levels['s2_max'],20))
                }]

    clf = GridSearchCV(FusedLassoEstimator(lasso_grid,fused_grid), two_d_grid,
                            scoring= 'neg_mean_squared_error',
                            n_jobs = -1, iid= False, refit=True,
                            cv=None, verbose=0, pre_dispatch='2*n_jobs',
                            error_score='raise-deprecating',
                            return_train_score='warn')

    clf.fit(X, y)


    s1 = clf.best_params_ ["s1"]
    s2 = clf.best_params_["s2"]

    s_opt_container[0,k] = s1
    s_opt_container[1,k] = s2
    for index ,i in enumerate(block_levels["n"]):  # i =10 i = 1000 usw

        mean_x = np.zeros(block_levels["p"])
        cov_X = np.identity(block_levels["p"])
        X = np.random.multivariate_normal(mean_x, cov_X,block_levels["n"][k])

        for j in range(block_levels["num_simulations"]):

            eps = np.random.rand(block_levels["n"][k])

            y = np.matmul(X,beta)+eps

            beta_container[:,j,index] = fused_lasso_primal(y,X,s1,s2)

beta_container_block_levels = beta_container
beta_block_level = beta
s_opt_container_block_levels = s_opt_container



plt.plot(beta_block_level)
plt.plot(beta_container_block_levels[:,1,1])
beta[80:100]
fig, axes = plt.subplots(2, 2)

axes[0,0].set_title('center')
axes[0, 0].hist(beta_container_block_levels[87,:,0])

axes[1,0].set_xlabel('zero')
axes[1, 0].hist(beta_container_block_levels[150,:,0])

axes[0, 1].hist(beta_container_block_levels[81,:,0])
axes[0,1].set_title('block_in')

axes[1, 1].hist(beta_container_block_levels[80,:,0])
axes[1,1].set_xlabel('block_out')



        #block_spikes setting auch wenn über all block levels steht!!
    block_levels ={
        "n": [10,100],
        "p": 200,
        "number_of_blocks": 3,
        "amplitude": 3,
        "length_blocks": 10,
        "spike_level": 6,
        "spikes": 3,
        "levels": 1,
        "s1_min": 40,
        "s1_max": 200,
        "s2_min": 3,
        "s2_max": 100,
        "num_simulations": 200
    }

num_simulations = block_levels['num_simulations']
p = block_levels['p']
n = block_levels['n']
number_blocks = block_levels['number_of_blocks']
length_blocks = block_levels['length_blocks']
amplitude = block_levels['amplitude']
spike_level = block_levels['spike_level']
levels = block_levels['levels']
spikes = block_levels['spikes']

beta = generate_blocks(p, number_blocks, length_blocks, amplitude,
                    spike_level,levels, spikes)



beta_container = np.ones((block_levels["p"], block_levels["num_simulations"], len(block_levels["n"])))
s_opt_container = np.zeros([2,len(block_levels["n"])])

for k in list(range(len(block_levels["n"]))):

    mean_x = np.zeros(block_levels["p"])
    cov_X = np.identity(block_levels["p"])
    X = np.random.multivariate_normal(mean_x, cov_X,block_levels["n"][k])

    eps = np.random.randn(block_levels["n"][k])

    y = X.dot(beta)+eps

    lasso_grid = {
      's1': list(np.linspace(block_levels['s1_min'],block_levels['s1_max'],20))
    }
    fused_grid = {
      's2': list(np.linspace(block_levels['s2_min'],block_levels['s2_max'],20))
    }

    two_d_grid = [{
                's1': list(np.linspace(block_levels['s1_min'],block_levels['s1_max'],20)),
                's2': list(np.linspace(block_levels['s2_min'],block_levels['s2_max'],20))
                }]

    clf = GridSearchCV(FusedLassoEstimator(lasso_grid,fused_grid), two_d_grid,
                            scoring= 'neg_mean_squared_error',
                            n_jobs = -1, iid= False, refit=True,
                            cv=None, verbose=0, pre_dispatch='2*n_jobs',
                            error_score='raise-deprecating',
                            return_train_score='warn')

    clf.fit(X, y)


    s1 = clf.best_params_ ["s1"]
    s2 = clf.best_params_["s2"]

    s_opt_container[0,k] = s1
    s_opt_container[1,k] = s2
    for index ,i in enumerate(block_levels["n"]):  # i =10 i = 1000 usw

        mean_x = np.zeros(block_levels["p"])
        cov_X = np.identity(block_levels["p"])
        X = np.random.multivariate_normal(mean_x, cov_X,block_levels["n"][k])

        for j in range(block_levels["num_simulations"]):

            eps = np.random.rand(block_levels["n"][k])

            y = np.matmul(X,beta)+eps

            beta_container[:,j,index] = fused_lasso_primal(y,X,s1,s2)

beta_container_block_spikes = beta_container
beta_block_spikes = beta
s_opt_container_block_spikes = s_opt_container



plt.plot(beta_block_spikes)
plt.plot(beta_container_block_spikes[:,1,1])
beta[50:60]
fig, axes = plt.subplots(2, 2)

axes[0,0].set_title('center')
axes[0, 0].hist(beta_container_block_spikes[125,:,1])

axes[1,0].set_xlabel('zero')
axes[1, 0].hist(beta_container_block_spikes[150,:,1])

axes[0, 1].hist(beta_container_block_spikes[51,:,1])
axes[0,1].set_title('block_in')

axes[1, 1].hist(beta_container_block_spikes[50,:,1])
axes[1,1].set_xlabel('block_out')

        #block_spikes setting auch wenn über all block levels steht!!
    block_levels ={
        "n": [10,100],
        "p": 200,
        "number_of_blocks": 3,
        "amplitude": 3,
        "length_blocks": 10,
        "spike_level": 6,
        "spikes": 3,
        "levels": 1,
        "s1_min": 40,
        "s1_max": 200,
        "s2_min": 3,
        "s2_max": 100,
        "num_simulations": 200
    }

num_simulations = block_levels['num_simulations']
p = block_levels['p']
n = block_levels['n']
number_blocks = block_levels['number_of_blocks']
length_blocks = block_levels['length_blocks']
amplitude = block_levels['amplitude']
spike_level = block_levels['spike_level']
levels = block_levels['levels']
spikes = block_levels['spikes']

beta = generate_blocks(p, number_blocks, length_blocks, amplitude,
                    spike_level,levels, spikes)



beta_container = np.ones((block_levels["p"], block_levels["num_simulations"], len(block_levels["n"])))
s_opt_container = np.zeros([2,len(block_levels["n"])])

for k in list(range(len(block_levels["n"]))):

    mean_x = np.zeros(block_levels["p"])
    cov_X = np.identity(block_levels["p"])
    X = np.random.multivariate_normal(mean_x, cov_X,block_levels["n"][k])

    eps = np.random.randn(block_levels["n"][k])

    y = X.dot(beta)+eps

    lasso_grid = {
      's1': list(np.linspace(block_levels['s1_min'],block_levels['s1_max'],20))
    }
    fused_grid = {
      's2': list(np.linspace(block_levels['s2_min'],block_levels['s2_max'],20))
    }

    two_d_grid = [{
                's1': list(np.linspace(block_levels['s1_min'],block_levels['s1_max'],20)),
                's2': list(np.linspace(block_levels['s2_min'],block_levels['s2_max'],20))
                }]

    clf = GridSearchCV(FusedLassoEstimator(lasso_grid,fused_grid), two_d_grid,
                            scoring= 'neg_mean_squared_error',
                            n_jobs = -1, iid= False, refit=True,
                            cv=None, verbose=0, pre_dispatch='2*n_jobs',
                            error_score='raise-deprecating',
                            return_train_score='warn')

    clf.fit(X, y)


    s1 = clf.best_params_ ["s1"]
    s2 = clf.best_params_["s2"]

    s_opt_container[0,k] = s1
    s_opt_container[1,k] = s2
    for index ,i in enumerate(block_levels["n"]):  # i =10 i = 1000 usw

        mean_x = np.zeros(block_levels["p"])
        cov_X = np.identity(block_levels["p"])
        X = np.random.multivariate_normal(mean_x, cov_X,block_levels["n"][k])

        for j in range(block_levels["num_simulations"]):

            eps = np.random.rand(block_levels["n"][k])

            y = np.matmul(X,beta)+eps

            beta_container[:,j,index] = fused_lasso_primal(y,X,s1,s2)

beta_container_block_spikes = beta_container
beta_block_spikes = beta
s_opt_container_block_spikes = s_opt_container



plt.plot(beta_block_spikes)
plt.plot(beta_container_block_spikes[:,1,1])
beta[50:60]
fig, axes = plt.subplots(2, 2)

axes[0,0].set_title('center')
axes[0, 0].hist(beta_container_block_spikes[125,:,1])

axes[1,0].set_xlabel('zero')
axes[1, 0].hist(beta_container_block_spikes[150,:,1])

axes[0, 1].hist(beta_container_block_spikes[51,:,1])
axes[0,1].set_title('block_in')

axes[1, 1].hist(beta_container_block_spikes[50,:,1])
axes[1,1].set_xlabel('block_out')


###block setting
block_levels ={
    "n": [10,100],
    "p": 200,
    "number_of_blocks": 3,
    "amplitude": 3,
    "length_blocks": 10,
    "spike_level": 0,
    "spikes": 0,
    "levels": 0,
    "s1_min": 40,
    "s1_max": 200,
    "s2_min": 3,
    "s2_max": 100,
    "num_simulations": 200
}

num_simulations = block_levels['num_simulations']
p = block_levels['p']
n = block_levels['n']
number_blocks = block_levels['number_of_blocks']
length_blocks = block_levels['length_blocks']
amplitude = block_levels['amplitude']
spike_level = block_levels['spike_level']
levels = block_levels['levels']
spikes = block_levels['spikes']

beta = generate_blocks(p, number_blocks, length_blocks, amplitude,
                spike_level,levels, spikes)



beta_container = np.ones((block_levels["p"], block_levels["num_simulations"], len(block_levels["n"])))
s_opt_container = np.zeros([2,len(block_levels["n"])])

for k in list(range(len(block_levels["n"]))):

mean_x = np.zeros(block_levels["p"])
cov_X = np.identity(block_levels["p"])
X = np.random.multivariate_normal(mean_x, cov_X,block_levels["n"][k])

eps = np.random.randn(block_levels["n"][k])

y = X.dot(beta)+eps

lasso_grid = {
  's1': list(np.linspace(block_levels['s1_min'],block_levels['s1_max'],20))
}
fused_grid = {
  's2': list(np.linspace(block_levels['s2_min'],block_levels['s2_max'],20))
}

two_d_grid = [{
            's1': list(np.linspace(block_levels['s1_min'],block_levels['s1_max'],20)),
            's2': list(np.linspace(block_levels['s2_min'],block_levels['s2_max'],20))
            }]

clf = GridSearchCV(FusedLassoEstimator(lasso_grid,fused_grid), two_d_grid,
                        scoring= 'neg_mean_squared_error',
                        n_jobs = -1, iid= False, refit=True,
                        cv=None, verbose=0, pre_dispatch='2*n_jobs',
                        error_score='raise-deprecating',
                        return_train_score='warn')

clf.fit(X, y)


s1 = clf.best_params_ ["s1"]
s2 = clf.best_params_["s2"]

s_opt_container[0,k] = s1
s_opt_container[1,k] = s2
for index ,i in enumerate(block_levels["n"]):  # i =10 i = 1000 usw

    mean_x = np.zeros(block_levels["p"])
    cov_X = np.identity(block_levels["p"])
    X = np.random.multivariate_normal(mean_x, cov_X,block_levels["n"][k])

    for j in range(block_levels["num_simulations"]):

        eps = np.random.rand(block_levels["n"][k])

        y = np.matmul(X,beta)+eps

        beta_container[:,j,index] = fused_lasso_primal(y,X,s1,s2)

beta_container_blocks = beta_container
beta_blocks = beta
s_opt_container_blocks = s_opt_container



plt.plot(beta_blocks)
plt.plot(beta_container_blocks[:,1,1])
beta[50:60]
fig, axes = plt.subplots(2, 2)

axes[0,0].set_title('center')
axes[0, 0].hist(beta_container_blocks[125,:,1])

axes[1,0].set_xlabel('zero')
axes[1, 0].hist(beta_container_blocks[150,:,1])

axes[0, 1].hist(beta_container_blocks[51,:,1])
axes[0,1].set_title('block_in')

axes[1, 1].hist(beta_container_blocks[50,:,1])
axes[1,1].set_xlabel('block_out')
