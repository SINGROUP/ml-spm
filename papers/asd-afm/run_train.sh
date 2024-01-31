#!/bin/bash

# Choose dataset to use for training ('light' or 'heavy'). Needs to be generated first using generate_data.py.
DATASET='heavy'

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
        --run_dir ./train_$DATASET \
        --data_dir ./data_$DATASET \
        --urls_train "data_$DATASET-K-0_train_{0..15}.tar" \
        --urls_val "data_$DATASET-K-0_val_{0..15}.tar" \
        --urls_test "data_$DATASET-K-0_test_{0..15}.tar" \
        --random_seed 0 \
        --train True \
        --test True \
        --predict True \
        --epochs 50 \
        --num_workers $N_WORKERS \
        --batch_size $BATCH_SIZE \
        --avg_best_epochs 3 \
        --pred_batches 3 \
        --lr 1e-3 \
        --loss_labels "Atomic Disks" "vdW Spheres" "Height Map" \
        --loss_weights 20.0 0.2 0.1 \
        --timings
