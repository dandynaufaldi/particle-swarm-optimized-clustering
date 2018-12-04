import numpy as np
import pandas as pd

from pso import ParticleSwarmOptimizedClustering
from utils import normalize
if __name__ == "__main__":
    data = pd.read_csv('seed.txt', sep='\t', header=None)
    x = data.drop([7], axis=1)
    # print(x.head())
    x = x.values
    x = normalize(x)
    pso = ParticleSwarmOptimizedClustering(
        n_cluster=3, n_particles=10, data=x, hybrid=True)  #, max_iter=2000, print_debug=50)
    pso.run()
