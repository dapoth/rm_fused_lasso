"""Draw simulated samples from two uncorrelated uniform variables
(locations in two dimensions) for two types of agents and store
them in a 3-dimensional NumPy array.

*Note:* In principle, one would read the number of dimensions etc.
from the "IN_MODEL_SPECS" file, this is to demonstrate the most basic
use of *run_py_script* only.

"""

import numpy as np
from bld.project_paths import project_paths_join as ppj


np.random.seed(12345)

n = 100
p = 10


def generate_data():
    beta = np.zeros(p)
    eps = np.random.randn(n)
    mean = np.ones(p)
    cov = np.identity(p)
    x = np.random.multivariate_normal(mean, cov,n)
    y = np.matmul(x,beta)+eps
    return y,x,beta,eps


def save_y(y):
    y.tofile(ppj("OUT_DATA", "y.csv"), sep=",")
    #np.savetxt(ppj("OUT_DATA", "y.csv"),y)
def save_x(x):
    x.tofile(ppj("OUT_DATA", "x.csv"), sep=",")
def save_beta(beta):
    beta.tofile(ppj("OUT_DATA", "beta.csv"), sep=",")
def save_eps(eps):
    eps.tofile(ppj("OUT_DATA", "eps.csv"), sep=",")    
    
    
    


if __name__ == "__main__":
    y,x,beta,eps = generate_data()
    save_y(y)
    save_x(x)
    save_beta(beta)
    save_eps(eps)
