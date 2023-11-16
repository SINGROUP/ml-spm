#!/bin/bash

export OMP_NUM_THREADS=1

# Unlike for the PosNet training, the GraphImgNet at the moment only supports single-GPU training, so we don't use torchrun here.
# Slightly different parameters where used with the different datasets for training the models in the paper:
# - AFM-ice-Cu111:
#   --dataset AFM-ice-Cu111
#   --data_dir ./dataset_Cu111
#   --urls_train "Ice-K-{1..10}_train_{0..31}.tar"
#   --urls_val "Ice-K-{1..10}_val_{0..7}.tar"
#   --urls_test "Ice-K-{1..10}_test_{0..7}.tar"
#   --zmin -2.5
#   --z_lims -2.9 0.5
#   --epochs 1000
#   --batch_size 16
# - AFM-ice-Au111-monolayer:
#   --dataset AFM-ice-Au111-monolayer
#   --data_dir ./dataset_Au111-monolayer
#   --urls_train "Ice-K-{1..10}_train_{0..31}.tar"
#   --urls_val "Ice-K-{1..8}_val_{0..7}.tar"
#   --urls_test "Ice-K-{1..10}_test_{0..7}.tar"
#   --zmin -2.5
#   --z_lims -2.9 0.5
#   --epochs 1000
#   --batch_size 16
python fit_graphnet.py \
    --run_dir ./fit_graphnet_Cu111 \
    --dataset AFM-ice-Au111-bilayer \
    --data_dir ./dataset_Au111-bilayer \
    --urls_train "Ice-K-{1..10}_train_{0..31}.tar" \
    --urls_val "Ice-K-{1..8}_val_{0..3}.tar" \
    --urls_test "Ice-K-{1..10}_test_{0..3}.tar" \
    --random_seed 0 \
    --train True \
    --test True \
    --predict True \
    --epochs 250 \
    --num_workers 6 \
    --batch_size 32 \
    --avg_best_epochs 10 \
    --pred_batches 3 \
    --lr 1e-3 \
    --zmin -3.5 \
    --z_lims -3.5 0.5 \
    --peak_std 0.20 \
    --box_res 0.125 0.125 0.100 \
    --classes 1 8 \
    --class_colors 'w' 'r' \
    --edge_cutoff 3.0 \
    --afm_cutoff 1.125 \
    --loss_weights 1.0 1.0 \
    --loss_labels "NLL (Node)" "NLL (Edge)" \
    --timings
