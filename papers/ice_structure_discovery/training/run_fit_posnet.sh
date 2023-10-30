#!/bin/bash

export OMP_NUM_THREADS=1

torchrun \
    --standalone \
    --nnodes 1 \
    --nproc_per_node 1 \
    --max_restarts 0 \
    fit_posnet.py \
        --run_dir ./fit_posnet_Cu111 \
        --dataset AFM-ice-Cu111 \
        --data_dir ./dataset_Cu111 \
        --urls_train "Ice-K-{1..10}_train_{0..31}.tar" \
        --urls_val "Ice-K-{1..5}_val_{0..7}.tar" \
        --urls_test "Ice-K-{1..10}_test_{0..7}.tar" \
        --random_seed 0 \
        --train True \
        --test True \
        --predict True \
        --epochs 1000 \
        --num_workers 4 \
        --batch_size 8 \
        --avg_best_epochs 5 \
        --pred_batches 10 \
        --lr 1e-3 \
        --zmin -2.9 \
        --z_lims -3.4 0.5 \
        --peak_std 0.20 \
        --box_res 0.125 0.125 0.100 \
        --loss_labels "MSE (pos.)" \
        --timings
