#!/bin/bash

export OMP_NUM_THREADS=1

python fit_graphnet.py \
    --run_dir ./fit_graphnet \
    --random_seed 0 \
    --train True \
    --test True \
    --predict True \
    --epochs 20 \
    --num_workers 2 \
    --batch_size 2 \
    --avg_best_epochs 5 \
    --pred_batches 20 \
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
    --data_dir /mnt/triton_project/AFM_Hartree_DB/AFM_sims/striped/Xylose-Cu111 \
    --urls_train "Xylose-K-{1..2}_train_{0..1}.tar" \
    --urls_val "Xylose-K-{1..1}_val_{0..3}.tar" \
    --urls_test "Xylose-K-{1..2}_test_{0..3}.tar" \
    --timings
