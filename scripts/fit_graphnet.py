
import os
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import webdataset as wds
import yaml
from torch import optim

import sys
sys.path.append('/home/work/git/ml-spm')
import mlspm.data_loading as dl
import mlspm.preprocessing as pp
import mlspm.visualization as vis
from mlspm import graph, utils
from mlspm.cli import parse_args
from mlspm.logging import LossLogPlot, SyncedLoss
from mlspm.losses import GraphLoss
from mlspm.models import GraphImgNet, PosNet


def make_model(device, cfg):
    outsize = round((cfg['z_lims'][1] - cfg['z_lims'][0]) / cfg['box_res'][2]) + 1
    posnet = PosNet(
        encode_block_channels   = [16, 32, 64, 128],
        encode_block_depth      = 3,
        decode_block_channels   = [128, 64, 32],
        decode_block_depth      = 2,
        decode_block_channels2  = [128, 64, 32],
        decode_block_depth2     = 3,
        attention_channels      = [128, 128, 128],
        res_connections         = True,
        activation              = 'relu',
        padding_mode            = 'zeros',
        pool_type               = 'avg',
        decoder_z_sizes         = [5, 15, outsize],
        z_outs                  = [3, 3, 5, 10],
        afm_res                 = cfg['box_res'][0],
        grid_z_range            = cfg['z_lims'],
        peak_std                = cfg['peak_std']
    ).to(device)
    model = GraphImgNet(
        posnet,
        n_classes               = len(cfg['classes']),
        iters                   = 5,
        node_feature_size       = 40,
        message_size            = 40,
        message_hidden_size     = 196,
        edge_cutoff             = cfg['edge_cutoff'],
        afm_cutoff              = cfg['afm_cutoff'],
        conv_channels           = [12, 24, 48],
        conv_depth              = 2,
        node_out_hidden_size    = 196,
        edge_out_hidden_size    = 196,
        res_connections         = True,
        activation              = 'relu',
        padding_mode            = 'zeros',
        pool_type               = 'avg',
        device                  = device
    )
    for param in model.posnet.parameters():
        param.requires_grad = False
    criterion = GraphLoss(*cfg['loss_weights'])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg['lr'])
    lr_decay_rate = 1e-5
    lr_decay = optim.lr_scheduler.LambdaLR(optimizer, lambda b: 1.0/(1.0+lr_decay_rate*b))
    return model, criterion, optimizer, lr_decay

def apply_preprocessing(batch, cfg):

    box_res = cfg['box_res']
    z_lims = cfg['z_lims']
    zmin = cfg['zmin']
    classes = cfg['classes']

    X, atoms, scan_windows = [batch[k] for k in ['X', 'xyz', 'sw']]

    # Pick a random number of slices between 1 and all of them, and randomize start slice between 0-4
    nz_max = X[0].shape[-1]
    nz = random.choice(range(1, nz_max+1))
    z0 = random.choice(range(0, min(5, nz_max+1-nz)))
    X = [x[:, :, :, -nz:] for x in X] if z0 == 0 else [x[:, :, :, -(nz+z0):-z0] for x in X]

    atoms = [a[a[:, -1] != 29] for a in atoms]
    atoms = pp.top_atom_to_zero(atoms)
    xyz = atoms.copy()
    bonds = graph.find_bonds(atoms)
    mols = [graph.MoleculeGraph(a, b, classes=classes) for a, b in zip(atoms, bonds)]
    mols = graph.threshold_atoms_bonds(mols, zmin)
    mols = [m.randomize_positions(sigma=[0.08, 0.08, 0.08]) for m in mols]
    mols, sw = graph.shift_mols_window(mols, scan_windows[0])

    box_borders = (
        (0, 0, z_lims[0]),
        (box_res[0]*(X[0].shape[1] - 1), box_res[1]*(X[0].shape[2] - 1), z_lims[1])
    )

    pp.rand_shift_xy_trend(X, max_layer_shift=0.02, max_total_shift=0.04)
    X, mols, box_borders = graph.add_rotation_reflection_graph(X, mols, box_borders, num_rotations=1,
        reflections=True, crop=(128, 128), per_batch_item=True)
    pp.add_norm(X)
    pp.add_gradient(X, c=0.3)
    pp.add_noise(X, c=0.1, randomize_amplitude=True, normal_amplitude=True)
    pp.add_cutout(X, n_holes=5)

    X = X[0]
    
    return X, mols, xyz

def make_webDataloader(cfg, mode='train'):
    
    assert mode in ['train', 'val', 'test'], mode

    shard_list = dl.ShardList(
        cfg[f'urls_{mode}'],
        base_path=cfg['data_dir'],
        world_size=cfg['world_size'],
        rank=cfg['global_rank'],
        substitute_param=(mode == 'train'),
        log=Path(cfg['run_dir']) / 'shards.log'
    )

    dataset = wds.WebDataset(shard_list)
    dataset.pipeline.pop()
    if mode == 'train': dataset.append(wds.shuffle(10))          # Shuffle order of shards
    dataset.append(wds.tariterators.tarfile_to_samples())        # Gather files inside tar files into samples
    dataset.append(wds.split_by_worker)                          # Use a different subset of samples in shards in different workers
    if mode == 'train': dataset.append(wds.shuffle(100))         # Shuffle samples within a worker process
    dataset.append(wds.decode('pill', dl.decode_xyz))            # Decode image and xyz files
    dataset.append(dl.rotate_and_stack(reverse=False))           # Combine separate images into a stack, reverse=True only for QUAM dataset
    dataset.append(dl.batched(cfg['batch_size']))                # Gather samples into batches
    dataset = dataset.map(partial(apply_preprocessing, cfg=cfg)) # Preprocess batch

    dataloader = wds.WebLoader(
        dataset,
        num_workers=cfg['num_workers'],
        batch_size=None, # Batching is done in the WebDataset
        pin_memory=True,
        collate_fn=dl.collate_graph,
        persistent_workers=False
    )
    
    return dataset, dataloader

def batch_to_device(batch, device):
    X, pos, node_classes, edges, ref_graphs, xyz = batch
    X = X.to(device)
    pos = [p.to(device) for p in pos]
    node_classes = [n.to(device) for n in node_classes]
    edges = [e.to(device) for e in edges]
    return X, pos, node_classes, edges, ref_graphs, xyz

def run(cfg):

    # Running with a single device, so set all ranks to 0
    cfg['rank'] = 0
    cfg['global_rank'] = 0
    cfg['local_rank'] = 0

    device = 'cuda'

    start_time = time.perf_counter()

    # Create directories
    checkpoint_dir = os.path.join(cfg['run_dir'], 'Checkpoints/')
    if not os.path.exists(cfg['run_dir']):
        os.makedirs(cfg['run_dir'])
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Define model, optimizer, and loss
    model, criterion, optimizer, lr_decay = make_model(device, cfg)
    
    print(f'World size = {cfg["world_size"]}')
    print(f'Trainable parameters: {utils.count_parameters(model)}')

    # Setup checkpointing and load a checkpoint if available
    checkpointer = utils.Checkpointer(model, optimizer, additional_data={'lr_params': lr_decay},
        checkpoint_dir=checkpoint_dir, keep_last_epoch=True)
    init_epoch = checkpointer.epoch

    if (init_epoch == 1) and (pretrained_weights := cfg['load_weights']):
        print(f'Loading pretrained weights from {pretrained_weights}')
        utils.load_checkpoint(
            model,
            optimizer,
            pretrained_weights,
            additional_data={'lr_params': lr_decay}
        )
    
    # Setup logging
    log_file = open(os.path.join(cfg['run_dir'], 'batches.log'), 'a')
    loss_logger = LossLogPlot(
        log_path=os.path.join(cfg['run_dir'], 'loss_log.csv'),
        plot_path=os.path.join(cfg['run_dir'], 'loss_history.png'),
        loss_labels=cfg['loss_labels'],
        loss_weights=cfg['loss_weights'],
        print_interval=cfg['print_interval'],
        init_epoch=init_epoch,
        stream=log_file
    )
    
    if cfg['train']:

        # Create datasets and dataloaders
        _, train_loader = make_webDataloader(cfg, 'train')
        _, val_loader = make_webDataloader(cfg, 'val')

        if init_epoch <= cfg['epochs']:
            print(f'\n ========= Starting training from epoch {init_epoch}')
        else:
            print('Model already trained')
        
        for epoch in range(init_epoch, cfg['epochs']+1):

            print(f'\n === Epoch {epoch}')

            # Train
            if cfg['timings']:
                t0 = time.perf_counter()

            model.train()
            for ib, batch in enumerate(train_loader):

                # Transfer batch to device
                X, pos, node_classes_ref, edges_ref, _, _ = batch_to_device(batch, device)

                if cfg['timings']:
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                
                # Forward
                pred = model(X, pos)
                losses = criterion(pred, (node_classes_ref, edges_ref), separate_loss_factors=True)
                loss = losses[0]
                
                if cfg['timings']: 
                    torch.cuda.synchronize()
                    t2 = time.perf_counter()
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_decay.step()

                # Log losses
                loss_logger.add_train_loss(losses)

                if cfg['timings']:
                    torch.cuda.synchronize()
                    t3 = time.perf_counter()
                    print(f'(Train) Load Batch/Forward/Backward: {t1-t0:6f}/{t2-t1:6f}/{t3-t2:6f}')
                    t0 = t3

            # Validate
            val_start = time.perf_counter()
            if cfg['timings']: t0 = val_start
            
            model.eval()
            with torch.no_grad():
                
                for ib, batch in enumerate(val_loader):
                    
                    # Transfer batch to device
                    X, pos, node_classes_ref, edges_ref, _, _ = batch_to_device(batch, device)
                    
                    if cfg['timings']: 
                        torch.cuda.synchronize()
                        t1 = time.perf_counter()
                    
                    # Forward
                    pred = model(X, pos)
                    losses = criterion(pred, (node_classes_ref, edges_ref), separate_loss_factors=True)

                    loss_logger.add_val_loss(losses)
                    
                    if cfg['timings']:
                        torch.cuda.synchronize()
                        t2 = time.perf_counter()
                        print(f'(Val) Load Batch/Forward: {t1-t0:6f}/{t2-t1:6f}')
                        t0 = t2

            # Write average losses to log and report to terminal
            loss_logger.next_epoch()

            # Save checkpoint
            checkpointer.next_epoch(loss_logger.val_losses[-1][0])
            
    # Return to best epoch
    checkpointer.revert_to_best_epoch()

    # Load pretrained weights for Posnet from a separate file
    posnet_weights = torch.load(cfg['posnet_path'])
    model.posnet.load_state_dict(posnet_weights)

    # Save final best model weights
    torch.save(model.state_dict(), save_path := os.path.join(cfg['run_dir'], 'best_model.pth'))
    print(f'\nModel saved to {save_path}')

    print(f'Best validation loss on epoch {checkpointer.best_epoch}: {checkpointer.best_loss}')
    print(f'Average of best {cfg["avg_best_epochs"]} validation losses: '
        f'{np.sort(loss_logger.val_losses[:, 0])[:cfg["avg_best_epochs"]].mean()}')

    if cfg['test'] or cfg['predict']:
        _, test_loader = make_webDataloader(cfg, 'test')

    if cfg['test']:

        print(f'\n ========= Testing with model from epoch {checkpointer.best_epoch}')

        stats_ref_pos = graph.GraphStats(cfg['classes'])
        stats_pred_pos = graph.GraphStats(cfg['classes'])
        eval_losses = SyncedLoss(num_losses=len(cfg['loss_labels']))
        eval_start = time.perf_counter()
        if cfg['timings']: t0 = eval_start
        
        model.eval()
        with torch.no_grad():
            
            for ib, batch in enumerate(test_loader):
                
                # Transfer batch to device
                X, pos, node_classes_ref, edges_ref, ref_graphs, _ = batch_to_device(batch, device)
                
                if cfg['timings']:
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                
                # Forward
                pred = model(X, pos)
                losses = criterion(pred, (node_classes_ref, edges_ref), separate_loss_factors=True)
                eval_losses.append(losses)

                if cfg['timings']:
                    torch.cuda.synchronize()
                    t2 = time.perf_counter()

                # Gather statistical information
                pred_graphs_ref_pos = model.pred_to_graph(pos, *pred, bond_threshold=0.5)
                pred_graphs_pred_pos = model.predict_graph(X, bond_threshold=0.5)
                stats_ref_pos.add_batch(pred_graphs_ref_pos, ref_graphs)
                stats_pred_pos.add_batch(pred_graphs_pred_pos, ref_graphs)

                if (ib+1) % cfg['print_interval'] == 0: print(f'Test Batch {ib+1}')
                
                if cfg['timings']:
                    torch.cuda.synchronize()
                    t3 = time.perf_counter()
                    print(f'(Test) t0/Load Batch/Forward/Stats: {t1-t0:6f}/{t2-t1:6f}/{t3-t2:6f}')
                    t0 = t3

        # Save statistical information
        stats_dir1 = os.path.join(cfg['run_dir'], 'stats_ref_pos')
        stats_dir2 = os.path.join(cfg['run_dir'], 'stats_pred_pos')
        stats_ref_pos.plot(stats_dir1)
        stats_ref_pos.report(stats_dir1)
        stats_pred_pos.plot(stats_dir2)
        stats_pred_pos.report(stats_dir2)

        # Average losses and print
        eval_loss = eval_losses.mean()
        print(f'Test set loss: {loss_logger.loss_str(eval_loss)}')

        # Save test set loss to file
        with open(os.path.join(cfg['run_dir'], 'test_loss.txt'),'w') as f:
            f.write(';'.join([str(l) for l in eval_loss]))

    if cfg['predict']:
    
        # Make predictions
        print(f'\n ========= Predict on {cfg["pred_batches"]} batches from the test set')
        counter = 0
        pred_dir1 = os.path.join(cfg['run_dir'], 'predictions_ref_pos/')
        pred_dir2 = os.path.join(cfg['run_dir'], 'predictions_pred_pos/')
        
        with torch.no_grad():
            
            for ib, batch in enumerate(test_loader):
            
                if ib >= cfg['pred_batches']: break
                
                # Transfer batch to device
                X, pos, node_classes_ref, edges_ref, ref_graphs, xyz = batch_to_device(batch, device)
                
                # Forward
                pred = model(X, pos)
                losses = criterion(pred, (node_classes_ref, edges_ref), separate_loss_factors=True)
                pred_graphs_ref_pos = model.pred_to_graph(pos, *pred, bond_threshold=0.5)
                pred_graphs_pred_pos, grid_pred = model.predict_graph(X, bond_threshold=0.5, return_grid=True)
                grid_pred = grid_pred.cpu().numpy()

                # Save xyzs
                utils.batch_write_xyzs(xyz, outdir=pred_dir1, start_ind=counter)
                graph.save_graphs_to_xyzs(pred_graphs_ref_pos, cfg['classes'],
                    outfile_format=os.path.join(pred_dir1, '{ind}_graph_pred.xyz'), start_ind=counter)
                graph.save_graphs_to_xyzs(ref_graphs, cfg['classes'], 
                    outfile_format=os.path.join(pred_dir1, '{ind}_graph_ref.xyz'), start_ind=counter)
                utils.batch_write_xyzs(xyz, outdir=pred_dir2, start_ind=counter)
                graph.save_graphs_to_xyzs(pred_graphs_pred_pos, cfg['classes'],
                    outfile_format=os.path.join(pred_dir2, '{ind}_graph_pred.xyz'), start_ind=counter)
                graph.save_graphs_to_xyzs(ref_graphs, cfg['classes'], 
                    outfile_format=os.path.join(pred_dir2, '{ind}_graph_ref.xyz'), start_ind=counter)
            
                # Visualize predictions
                box_borders = model.posnet.make_box_borders(X.shape[1:3])
                grid_ref = graph.make_position_distribution(ref_graphs, box_borders, box_res=cfg['box_res'],
                    std = cfg['peak_std'])
                vis.plot_graphs(pred_graphs_ref_pos, ref_graphs, box_borders=box_borders,
                    outdir=pred_dir1, start_ind=counter, classes=cfg['classes'], class_colors=cfg['class_colors'])
                vis.plot_graphs(pred_graphs_pred_pos, ref_graphs, box_borders=box_borders,
                    outdir=pred_dir2, start_ind=counter, classes=cfg['classes'], class_colors=cfg['class_colors'])
                vis.plot_distribution_grid(grid_pred, grid_ref, box_borders=box_borders, outdir=pred_dir2,
                    start_ind=counter)

                # Plot input AFM images
                X = X.cpu().numpy()
                vis.make_input_plots([X], outdir=pred_dir1, start_ind=counter)
                vis.make_input_plots([X], outdir=pred_dir2, start_ind=counter)

                counter += len(X)

    print(f'Done. Total time: {time.perf_counter() - start_time:.0f}s')

    log_file.close()

if __name__ == '__main__':
    
    # Get config
    cfg = parse_args()
    run_dir = Path(cfg['run_dir'])
    run_dir.mkdir(exist_ok=True, parents=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        # Remember settings
        yaml.safe_dump(cfg, f)

    # Set random seeds
    torch.manual_seed(cfg['random_seed'])
    random.seed(cfg['random_seed'])
    np.random.seed(cfg['random_seed'])

    # Start run
    cfg['world_size'] = 1
    cfg['posnet_path'] = './posnet.pth'
    run(cfg)
