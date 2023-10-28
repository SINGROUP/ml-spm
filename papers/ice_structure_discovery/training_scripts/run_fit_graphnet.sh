#!/bin/bash

export OMP_NUM_THREADS=1

python fit_graphnet.py \
    --run_dir ./fit_graphnet_Cu111 \
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
    --num_workers 6 \
    --batch_size 32 \
    --avg_best_epochs 5 \
    --pred_batches 3 \
    --lr 1e-3 \
    --zmin -2.9 \
    --z_lims -3.4 0.5 \
    --peak_std 0.20 \
    --box_res 0.125 0.125 0.100 \
    --classes 1 6,14 7,15 8,16 9,17,35 \
    --class_colors 'w' 'dimgray' 'b' 'r' 'yellowgreen' \
    --edge_cutoff 3.0 \
    --afm_cutoff 1.125 \
    --loss_weights 1.0 1.0 \
    --loss_labels "NLL (Node)" "NLL (Edge)" \
    --timings
