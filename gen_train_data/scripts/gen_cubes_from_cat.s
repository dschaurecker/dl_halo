#!/bin/bash

#SBATCH --output=slurm_%j_gen_sim.out
#SBATCH --job-name=gen_sim
#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --time=12:00:00 #2hours for 2048^3 pixels
#SBATCH --cpus-per-task=1
#SBARCH --task-per-node=1


module purge


singularity exec \
	    --overlay /scratch/ds6311/nbodykit.ext3:ro \
	    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
	    bash -c "source /ext3/env.sh; python /scratch/ds6311/github/dl_halo/gen_train_data/codes/gen_cubes_from_cat.py \
        --simsize 75. --pixside 2048 --crop 64 --pad 24 --illustris '3' --window 'tsc' "


echo
echo "Done"
echo

#
