#!/bin/bash

# Number of GPUs and the number of samples per batch per GPU (total batch size = N_GPU x BATCH_SIZE).
# The original training used 32 GPUs divided over 4 nodes with one sample per GPU, but only a single
# node is used here for simplicity.
N_GPU=4
BATCH_SIZE=8

# Number of parallel workers per GPU for loading data from disk.
N_WORKERS=4

export OMP_NUM_THREADS=1

# Slightly different parameters where used with the different datasets for training the models in the paper:
# - AFM-ice-Cu111:
#   --dataset AFM-ice-Cu111
#   --data_dir ./dataset_Cu111
#   --urls_train "Ice-K-{1..10}_train_{0..31}.tar"
#   --urls_val "Ice-K-{1..10}_val_{0..7}.tar"
#   --urls_test "Ice-K-{1..10}_test_{0..7}.tar"
#   --z_lims -2.9 0.5
#   --epochs 1200
# - AFM-ice-Au111-monolayer:
#   --dataset AFM-ice-Au111-monolayer
#   --data_dir ./dataset_Au111-monolayer
#   --urls_train "Ice-K-{1..10}_train_{0..31}.tar"
#   --urls_val "Ice-K-{1..5}_val_{0..7}.tar"
#   --urls_test "Ice-K-{1..10}_test_{0..7}.tar"
#   --z_lims -2.9 0.5
#   --epochs 1000
torchrun \
    --standalone \
    --nnodes 1 \
    --nproc_per_node $N_GPU \
    --max_restarts 0 \
    fit_posnet.py \
        --run_dir ./fit_posnet_Cu111 \
        --dataset AFM-ice-Au111-bilayer \
        --data_dir ./dataset_Au111-bilayer \
        --urls_train "Ice-K-{1..10}_train_{0..31}.tar" \
        --urls_val "Ice-K-{1..5}_val_{0..3}.tar" \
        --urls_test "Ice-K-{1..10}_test_{0..3}.tar" \
        --random_seed 0 \
        --train True \
        --test True \
        --predict True \
        --epochs 800 \
        --num_workers $N_WORKERS \
        --batch_size $BATCH_SIZE \
        --avg_best_epochs 5 \
        --pred_batches 10 \
        --lr 1e-4 \
        --zmin -10.0 \
        --z_lims -3.5 0.5 \
        --peak_std 0.20 \
        --box_res 0.125 0.125 0.100 \
        --loss_labels "MSE (pos.)" \
        --timings
