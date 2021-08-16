#!/bin/bash

#SBATCH --output=slurm_%j_test_2048.out
#SBATCH --job-name=test_2048_64
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=70GB
#SBATCH --time=1:00:00


module purge

run='2021_May_16_64061__pod_vgan'
state='599'
    
# getenv("SLURM_CPUS_PER_TASK")

export OMP_NUN_THREADS=1

srun singularity exec \
	    --overlay /scratch/ds6311/nbodykit.ext3:ro \
	    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
	    bash -c "source /ext3/env.sh; python /scratch/$USER/github/dl_halo/train_test/scripts/m2m.py test \
     --in-folder '/scratch/ds6311/Illustris-3/dm_only_2048_counts_arrays_subcubes_64_pad24/tsc_noArtnew/cube7*/*/*' \
     --tgt-folder '/scratch/ds6311/Illustris-2/dm_only_2048_counts_arrays_subcubes_64_pad0/tsc_noArtnew/cube7*/*/*' \
     --in-norms 'mytorch.mylog1p' --tgt-norms 'mytorch.mylog1p' \
     --pad 0 \
     --model G \
     --batches 1 --loader-workers 1 \
     --load-state '/scratch/ds6311/mywork/scripts/states/'"$run"'/state_'"$state"'.pt' \
     --state-num $state \
     --train-run-name $run "

echo
echo "Done"
echo

# Leave a few empty lines in the end to avoid occasional EOF trouble.
