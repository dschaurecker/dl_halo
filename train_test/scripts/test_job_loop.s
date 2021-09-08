#!/bin/bash

#SBATCH --output=slurm_test_loop_%j.out
#SBATCH --job-name=test_loop
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=70GB
#SBATCH --time=6:00:00

module purge


run='your_run_name'

for i in {600..610}
do
    echo "testing state $i"

    state="$i"

    export OMP_NUN_THREADS=1

    srun singularity exec \
            --overlay /scratch/ds6311/nbodykit.ext3:ro \
            /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
            bash -c "source /ext3/env.sh; python /scratch/$USER/github/dl_halo/train_test/scripts/m2m.py test \
         --in-folder '/scratch/ds6311/github/dl_halo/train_test/dummy_data/test/low-res/*' \
         --tgt-folder '/scratch/ds6311/github/dl_halo/train_test/dummy_data/test/high-res/*' \
         --in-norms 'mytorch.mylog1p' --tgt-norms 'mytorch.mylog1p' \
         --pad 0 \
         --model G \
         --batches 1 --loader-workers 1 \
         --load-state '/scratch/ds6311/github/dl_halo/train_test/scripts/states/'"$run"'/state_'"$state"'.pt' \
         --state-num $state \
         --train-run-name $run "
         

echo
echo "Done"
echo

#
