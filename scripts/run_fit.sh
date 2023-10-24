#!/bin/bash

export OMP_NUM_THREADS=1

torchrun \
    --standalone \
    --nnodes 1 \
    --nproc_per_node 1 \
    --max_restarts 0 \
    fit_posnet.py \
        --run_dir ./test_posnet \
        --random_seed 0 \
        --train True \
        --test True \
        --predict True \
        --epochs 10 \
        --num_workers 2 \
        --batch_size 2 \
        --avg_best_epochs 5 \
        --pred_batches 20 \
        --lr 1e-3 \
        --zmin -1.0 \
        --z_lims -1.4 0.5 \
        --peak_std 0.20 \
        --box_res 0.125 0.125 0.100 \
        --data_dir /mnt/triton_project/AFM_Hartree_DB/AFM_sims/striped/Xylose-Cu111 \
        --urls_train "Xylose-K-{1..2}_train_{0..1}.tar" \
        --urls_val "Xylose-K-{1..1}_val_{0..3}.tar" \
        --urls_test "Xylose-K-{1..2}_test_{0..3}.tar" \
        --timings
