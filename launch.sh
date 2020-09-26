#!/usr/bin/bash
#SBATCH --time=00:15:00
#SBATCH -p cca
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

julia slurm.jl 
