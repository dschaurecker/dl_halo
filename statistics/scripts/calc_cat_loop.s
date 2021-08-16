#!/bin/bash

#SBATCH --output=slurm_%j_calc_cat_loop.out
#SBATCH --job-name=calc_cat_loop
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=50GB
#SBATCH --time=1:00:00


module purge

runname="your_testjob_run_name"
cubesz=64

for i in {590..600}
do
    echo "calc cat of state $i"

    statenum=$i
    
    singularity exec \
            --overlay /scratch/$USER/nbodykit.ext3:ro \
            /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
            bash -c "source /ext3/env.sh; python /scratch/$USER/github/dl_halo/statistics/codes/clac_cat.py \
        --run-name '$runname' --state-num $statenum \
        --counts  --pixside 2048 --cubesz $cubesz "
    echo
    echo "launching eval job of state $i"
    echo
    
    sbatch eval_cat.s $runname $statenum $cubesz
done


    
echo
echo "Done"
echo

#