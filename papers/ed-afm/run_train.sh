#!/bin/bash

# Number of GPUs and the number of samples per batch per GPU (total batch size = N_GPU x BATCH_SIZE).
N_GPU=1
BATCH_SIZE=30

# Number of parallel workers per GPU for loading data from disk.
N_WORKERS=8

export OMP_NUM_THREADS=1

torchrun \
    --standalone \
    --nnodes 1 \
    --nproc_per_node $N_GPU \
    --max_restarts 0 \
    train.py \
        --run_dir ./train \
        --data_dir ./data \
        --urls_train "data-K-0_train_{0..7}.tar" \
        --urls_val "data-K-0_val_{0..7}.tar" \
        --urls_test "data-K-0_test_{0..7}.tar" \
        --random_seed 0 \
        --train True \
        --test True \
        --predict True \
        --epochs 50 \
        --num_workers $N_WORKERS \
        --batch_size $BATCH_SIZE \
        --avg_best_epochs 3 \
        --pred_batches 3 \
        --lr 1e-4 \
        --loss_labels "ES" \
        --loss_weights 1.0 \
        --timings
