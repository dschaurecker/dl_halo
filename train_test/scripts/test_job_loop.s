#!/bin/bash

#SBATCH --output=slurm_%j_test_2048_loop.out
#SBATCH --job-name=test_2048_64_loop
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=70GB
#SBATCH --time=6:00:00

module purge


run='2021_May_29_13284__pod_vgan'

for i in {600..610}
do
    echo "testing state $i"

    state="$i"

    # getenv("SLURM_CPUS_PER_TASK")

    export OMP_NUN_THREADS=1

    srun singularity exec \
            --overlay /scratch/ds6311/nbodykit.ext3:ro \
            /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
            bash -c "source /ext3/env.sh; python /scratch/$USER/github/dl_halo/train_test/scripts/m2m.py test2048 \
         --in-folder '/scratch/ds6311/Illustris-3/dm_only_2048_counts_arrays_subcubes_64_pad24/tsc_noArtnew/cube7_1024_1024_1024/*/*' \
         --tgt-folder '/scratch/ds6311/Illustris-2/dm_only_2048_counts_arrays_subcubes_64_pad0/tsc_noArtnew/cube7_1024_1024_1024/*/*' \
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

#
