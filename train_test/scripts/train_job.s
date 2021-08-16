#!/bin/bash

#SBATCH --output=slurm_%j.out
#SBATCH --job-name=2048_vgan_64
#SBATCH --nodes=8
#SBATCH --mem=100GB
#SBATCH --time=48:00:00
#SBARCH --task-per-node=1
#SBATCH --gres=gpu:2 -c8

export MASTER_ADDR=$(hostname -s)-ib0
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)

rand=$(shuf -i 10000-65500 -n 1)

module purge

srun singularity exec --nv \
	    --overlay /scratch/ds6311/nbodykit.ext3:ro \
	    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
	    bash -c "source /ext3/env.sh; python /scratch/$USER/github/dl_halo/train_test/scripts/m2m.py train \
    --train-in-patterns '/scratch/ds6311/Illustris-3/dm_only_2048_counts_arrays_subcubes_64_pad24/tsc_noArtnew/cube[0-5]*/*/*' \
    --train-tgt-patterns '/scratch/ds6311/Illustris-2/dm_only_2048_counts_arrays_subcubes_64_pad0/tsc_noArtnew/cube[0-5]*/*/*' \
    --val-in-patterns '/scratch/ds6311/Illustris-3/dm_only_2048_counts_arrays_subcubes_64_pad24/tsc_noArtnew/cube6*/*/*' \
    --val-tgt-patterns '/scratch/ds6311/Illustris-2/dm_only_2048_counts_arrays_subcubes_64_pad0/tsc_noArtnew/cube6*/*/*' \
    --in-norms 'mytorch.mylog1p' --tgt-norms 'mytorch.mylog1p' \
    --pad 0 --scale-factor 1 \
    --model G \
    --adv-model D --cgan --percentile 1. --adv-r1-reg-interval 16 \
    --lr 5e-5 --adv-lr 1e-5 --batches 1 --loader-workers 4 \
    --epochs 2000 --seed 42 --adv-start 1 --incr-adv-lr 1. --randnumber '"$rand"' \
    --optimizer-args '{\"betas\": [0., 0.9], \"weight_decay\": 1e-4}' --optimizer AdamW --augment "


#
