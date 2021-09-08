#!/bin/bash

#SBATCH --output=slurm_train_dummy_%j.out
#SBATCH --job-name=train_dummy
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --time=12:00:00
#SBARCH --task-per-node=1
#SBATCH --gres=gpu:1 -c4

export MASTER_ADDR=$(hostname -s)-ib0
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)

rand=$(shuf -i 10000-65500 -n 1)

module purge

srun singularity exec --nv \
	    --overlay /scratch/ds6311/nbodykit.ext3:ro \
	    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
	    bash -c "source /ext3/env.sh; python /scratch/$USER/github/dl_halo/train_test/scripts/m2m.py train \
    --train-in-patterns '/scratch/ds6311/github/dl_halo/train_test/dummy_data/train/low-res/train/*' \
    --train-tgt-patterns '/scratch/ds6311/github/dl_halo/train_test/dummy_data/train/high-res/train/*' \
    --val-in-patterns '/scratch/ds6311/github/dl_halo/train_test/dummy_data/train/low-res/val/*' \
    --val-tgt-patterns '/scratch/ds6311/github/dl_halo/train_test/dummy_data/train/high-res/val/*' \
    --in-norms 'mytorch.mylog1p' --tgt-norms 'mytorch.mylog1p' \
    --pad 0 --scale-factor 1 \
    --model G \
    --adv-model D --cgan --percentile 1. --adv-r1-reg-interval 16 \
    --lr 5e-5 --adv-lr 1e-5 --batches 1 --loader-workers 4 \
    --epochs 5 --seed 42 --adv-start 1 --incr-adv-lr 1. --randnumber '"$rand"' \
    --optimizer-args '{\"betas\": [0., 0.9], \"weight_decay\": 1e-4}' --optimizer AdamW --augment "


#
