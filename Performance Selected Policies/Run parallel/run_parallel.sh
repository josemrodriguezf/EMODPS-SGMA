#!/bin/bash
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=24
#SBATCH --partition compute
#SBATCH --constraint=ib
#SBATCH -J py-wrapper2
#SBATCH -o SGMA2.out
#SBATCH -e SGMA2.err
#SBATCH --export=ALL
#SBATCH --time=5-00:00:00


module load openmpi-4.0/gcc
python3 --version
mpirun python3 Insights_Parallel_1.py
