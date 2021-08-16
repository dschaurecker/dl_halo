#!/bin/bash

#SBATCH --output=slurm_%j_2pcf.out
#SBATCH --job-name=2pcf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=550GB 
#SBATCH --time=48:00:00

module purge
module load openmpi/gcc/4.0.5

mpiexec -n 40 singularity exec \
            --overlay /scratch/$USER/nbodykit.ext3:ro \
            /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
            bash -c "source /ext3/env.sh; python /scratch/$USER/github/dl_halo/statistics/codes/2pcf.py "

# 