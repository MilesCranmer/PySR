import os
import numpy as np
from pysr import pysr

X = np.random.randn(100, 5)
y = (np.cos(X[:, 0]*10.0)*3.3/(X[:, 3]**2 + 2) + np.exp(X[:, 1]))/(np.sin(X[:, 2]) + 5)

nodes=4
cores=20

eqn_file = pysr(X, y,
        procs=cores, cluster_nodes=nodes,
        populations=cores*nodes*2,
        binary_operators='mult plus div'.split(' '),
        unary_operators='cos exp sin'.split(' '),
        niterations=1000,
        ncyclesperiteration=1000,
        slurm_cluster=True)

launch_file = """#!/usr/bin/bash
#SBATCH --time=01:00:00
#SBATCH -p cca
#SBATCH --nodes="""f"{nodes}""""
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task="""f"{cores}""""

julia """f"{eqn_file}"

launch_filename = f'{eqn_file[:-3]}.sh'

with open(launch_filename, 'w') as f:
    print(launch_file, file=f)

os.system(f"sbatch {launch_filename}")
