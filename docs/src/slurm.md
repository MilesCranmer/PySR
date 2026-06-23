# Slurm (Multi-node)

PySR supports running across multiple nodes on Slurm via `cluster_manager="slurm"`.
This backend is **allocation-based**: you request resources with Slurm (`sbatch`/`salloc`), then PySR launches Julia workers inside that allocation (using `SlurmClusterManager.jl`).

Here is a minimal `sbatch` example using 3 workers on each of 2 nodes (6 workers total).

Save this as `pysr_job.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=pysr
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=3
#SBATCH --time=01:00:00

set -euo pipefail
python pysr_script.py
```

Save this as `pysr_script.py` in the same directory:

```python
import numpy as np
from pysr import PySRRegressor

X = np.random.RandomState(0).randn(1000, 2)
y = X[:, 0] + 2 * X[:, 1]

model = PySRRegressor(
    niterations=200,
    populations=2,
    parallelism="multiprocessing",
    cluster_manager="slurm",
    procs=6,  # must match the Slurm allocation's total task count
)
model.fit(X, y)
print(model)
```

Submit it with:

```bash
sbatch pysr_job.sh
```

## Notes

- `procs` is the number of Julia worker processes. It must match the Slurm allocation's total tasks (e.g., `--ntasks` or `--nodes * --ntasks-per-node`).
- Run the Python script once (as the master) inside the allocation; do not wrap it in `srun`.
