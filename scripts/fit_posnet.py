
import os
import pickle
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import webdataset as wds
import yaml
from torch import nn, optim
from torch.distributed.algorithms.join import Join
from torch.nn.parallel import DistributedDataParallel

import sys
sys.path.append('/home/work/git/ml-spm')
import mlspm.data_loading as dl
import mlspm.preprocessing as pp
import mlspm.visualization as vis
from mlspm import graph, utils
from mlspm.cli import parse_args
from mlspm.logging import LossLogPlot, SyncedLoss
from mlspm.models import PosNet


def make_model(device, cfg):
    outsize = round((cfg['z_lims'][1] - cfg['z_lims'][0]) / cfg['box_res'][2]) + 1
    model = PosNet(
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
        peak_std                = cfg['peak_std']
    ).to(device)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    lr_decay_rate = 1e-5
    lr_decay = optim.lr_scheduler.LambdaLR(optimizer, lambda b: 1.0/(1.0+lr_decay_rate*b))
    return model, criterion, optimizer, lr_decay

def apply_preprocessing(batch, cfg):

    box_res = cfg['box_res']
    z_lims = cfg['z_lims']
    zmin = cfg['zmin']
    peak_std = cfg['peak_std']

    X, atoms, scan_windows = [batch[k] for k in ['X', 'xyz', 'sw']]

    # Pick a random number of slices between 1 and all of them, and randomize start slice between 0-4
    nz_max = X[0].shape[-1]
    nz = random.choice(range(1, nz_max+1))
    z0 = random.choice(range(0, min(5, nz_max+1-nz)))
    X = [x[:, :, :, -nz:] for x in X] if z0 == 0 else [x[:, :, :, -(nz+z0):-z0] for x in X]

    # atoms = [a[a[:, -1] != 29] for a in atoms]
    pp.top_atom_to_zero(atoms)
    xyz = atoms.copy()
    mols = [graph.MoleculeGraph(a, []) for a in atoms]
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
    
    mols = graph.threshold_atoms_bonds(mols, zmin)
    ref = graph.make_position_distribution(mols, box_borders, box_res=box_res, std=peak_std)

    return X, [ref], xyz, box_borders

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
        collate_fn=dl.default_collate,
        persistent_workers=False
    )
    
    return dataset, dataloader

def batch_to_device(batch, device):
    X, ref, *rest = batch
    X = X[0].to(device)
    ref = ref[0].to(device)
    return X, ref, *rest

def batch_to_host(batch):
    X, ref, pred, xyz = batch
    X = X.squeeze(1).cpu()
    ref = ref.cpu()
    pred = pred.cpu()
    return X, ref, pred, xyz

def run(cfg):

    print(f'Starting on global rank {cfg["global_rank"]}, local rank {cfg["local_rank"]}\n', flush=True)

    # Initialize the distributed environment.
    dist.init_process_group(cfg['comm_backend'])

    start_time = time.perf_counter()

    if cfg['global_rank'] == 0:
        # Create run directory
        if not os.path.exists(cfg['run_dir']):
            os.makedirs(cfg['run_dir'])
    dist.barrier()
    
    # Define model, optimizer, and loss
    model, criterion, optimizer, lr_decay = make_model(cfg['local_rank'], cfg)

    # For mixed precision training
    use_amp = False
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    if cfg['global_rank'] == 0:
        print(f'World size = {cfg["world_size"]}')
        print(f'Trainable parameters: {utils.count_parameters(model)}')

    # Setup checkpointing and load a checkpoint if available
    checkpointer = utils.Checkpointer(model, optimizer, additional_data={'lr_params': lr_decay, 'scaler': scaler},
        checkpoint_dir=os.path.join(cfg['run_dir'], 'Checkpoints/'), keep_last_epoch=True)
    init_epoch = checkpointer.epoch

    if (init_epoch == 1) and (pretrained_weights := cfg['load_weights']):
        print(f'Loading pretrained weights from {pretrained_weights}')
        utils.load_checkpoint(
            model,
            optimizer,
            pretrained_weights,
            additional_data={'lr_params': lr_decay, 'scaler': scaler},
            rank=cfg['local_rank']
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

    # Wrap model in DistributedDataParallel.
    model = DistributedDataParallel(model, device_ids=[cfg['local_rank']], find_unused_parameters=False)

    if cfg['train']:

        if cfg['global_rank'] == 0:
            if init_epoch <= cfg['epochs']:
                print(f'\n ========= Starting training from epoch {init_epoch}')
            else:
                print('Model already trained')
        
        for epoch in range(init_epoch, cfg['epochs']+1):

            # # Adjust the zmin value depending on the epoch after the 10th epoch
            # # Gradual increase from zmin=-1.9 to -2.9 over 30 epochs
            # if epoch < 10:
            #     cfg['zmin'] = -1.9
            # elif 10 < epoch <= 40:
            #     cfg['zmin'] = -1.9 - (epoch - 10) / 30
            # else:
            #     cfg['zmin'] = -2.9

            # Create datasets and dataloaders
            train_set, train_loader = make_webDataloader(cfg, 'train')
            val_set, val_loader = make_webDataloader(cfg, 'val')

            if cfg['global_rank'] == 0: print(f'\n === Epoch {epoch}')

            # Train
            if cfg['timings'] and cfg['global_rank'] == 0: t0 = time.perf_counter()

            model.train()
            with Join([model, loss_logger.get_joinable('train')]):
                for ib, batch in enumerate(train_loader):

                    # Transfer batch to device
                    X, ref, _, _ = batch_to_device(batch, cfg['local_rank'])

                    if cfg['timings'] and cfg['global_rank'] == 0:
                        torch.cuda.synchronize()
                        t1 = time.perf_counter()
                    
                    # Forward
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                        pred = model(X)
                        loss = criterion(pred, ref)
                    
                    if cfg['timings'] and cfg['global_rank'] == 0: 
                        torch.cuda.synchronize()
                        t2 = time.perf_counter()
                    
                    # Backward
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    lr_decay.step()

                    # Log losses
                    try:
                        loss_logger.add_train_loss(loss)
                    except ValueError as e:
                        torch.save(model.module.state_dict(), save_path := os.path.join(cfg['run_dir'], 'debug_model.pth'))
                        with open('debug_data', 'wb') as f:
                            pickle.dump((X.cpu().numpy(), ref.cpu().numpy()), f)
                        print(f'Save debug data on rank {cfg["global_rank"]}')
                        raise e

                    if cfg['timings'] and cfg['global_rank'] == 0:
                        torch.cuda.synchronize()
                        t3 = time.perf_counter()
                        print(f'(Train {ib}) Load Batch/Forward/Backward: {t1-t0:6f}/{t2-t1:6f}/{t3-t2:6f}')
                        t0 = t3

            # Validate
            if cfg['global_rank'] == 0:
                val_start = time.perf_counter()
                if cfg['timings']: t0 = val_start
            
            model.eval()
            with Join([loss_logger.get_joinable('val')]):
                with torch.no_grad():
                    
                    for ib, batch in enumerate(val_loader):
                        
                        # Transfer batch to device
                        X, ref, _, _ = batch_to_device(batch, cfg['local_rank'])
                        
                        if cfg['timings'] and cfg['global_rank'] == 0: 
                            torch.cuda.synchronize()
                            t1 = time.perf_counter()
                        
                        # Forward
                        pred = model.module(X)
                        loss = criterion(pred, ref)

                        loss_logger.add_val_loss(loss)
                        
                        if cfg['timings'] and cfg['global_rank'] == 0:
                            torch.cuda.synchronize()
                            t2 = time.perf_counter()
                            print(f'(Val {ib}) Load Batch/Forward: {t1-t0:6f}/{t2-t1:6f}')
                            t0 = t2

            # Write average losses to log and report to terminal
            loss_logger.next_epoch()

            # Save checkpoint
            checkpointer.next_epoch(loss_logger.val_losses[-1][0])
            
    # Return to best epoch, and save model weights
    dist.barrier()
    checkpointer.revert_to_best_epoch()
    if cfg['global_rank'] == 0:
        torch.save(model.module.state_dict(), save_path := os.path.join(cfg['run_dir'], 'best_model.pth'))
        print(f'\nModel saved to {save_path}')
        print(f'Best validation loss on epoch {checkpointer.best_epoch}: {checkpointer.best_loss}')
        print(f'Average of best {cfg["avg_best_epochs"]} validation losses: '
            f'{np.sort(loss_logger.val_losses[:, 0])[:cfg["avg_best_epochs"]].mean()}')

    if cfg['test'] or cfg['predict']:
        test_set, test_loader = make_webDataloader(cfg, 'test')

    if cfg['test']:

        if cfg['global_rank'] == 0: print(f'\n ========= Testing with model from epoch {checkpointer.best_epoch}')

        eval_losses = SyncedLoss(len(loss_logger.loss_labels))
        eval_start = time.perf_counter()
        if cfg['timings'] and cfg['global_rank'] == 0:
            t0 = eval_start
        
        model.eval()
        with Join([eval_losses]):
            with torch.no_grad():
                
                for ib, batch in enumerate(test_loader):
                    
                    # Transfer batch to device
                    X, ref, _, _ = batch_to_device(batch, cfg['local_rank'])
                    
                    if cfg['timings'] and cfg['global_rank'] == 0:
                        torch.cuda.synchronize()
                        t1 = time.perf_counter()
                    
                    # Forward
                    pred = model(X)
                    loss = criterion(pred, ref)
                    eval_losses.append(loss)

                    if (ib+1) % cfg['print_interval'] == 0 and cfg['global_rank'] == 0:
                        print(f'Test Batch {ib+1}', file=log_file, flush=True)
                    
                    if cfg['timings'] and cfg['global_rank'] == 0:
                        torch.cuda.synchronize()
                        t2 = time.perf_counter()
                        print(f'(Test {ib}) t0/Load Batch/Forward: {t1-t0:6f}/{t2-t1:6f}')
                        t0 = t2

        if cfg['global_rank'] == 0:

            # Average losses and print
            eval_loss = eval_losses.mean()
            print(f'Test set loss: {loss_logger.loss_str(eval_loss)}')

            # Save test set loss to file
            with open(os.path.join(cfg['run_dir'], 'test_loss.txt'),'w') as f:
                f.write(';'.join([str(l) for l in eval_loss]))

    if cfg['predict'] and cfg['global_rank'] == 0:
    
        # Make predictions
        print(f'\n ========= Predict on {cfg["pred_batches"]} batches from the test set')
        counter = 0
        pred_dir = os.path.join(cfg['run_dir'], 'predictions/')
        
        with torch.no_grad():
            
            for ib, batch in enumerate(test_loader):
            
                if ib >= cfg['pred_batches']: break
                
                # Transfer batch to device
                X, ref, xyz, box_borders = batch_to_device(batch, cfg['local_rank'])
                
                # Forward
                pred = model.module(X)
                loss = criterion(pred, ref)

                # Back to host
                X, ref, pred, xyz = batch_to_host((X, ref, pred, xyz))

                # Save xyzs
                utils.batch_write_xyzs(xyz, outdir=pred_dir, start_ind=counter)
            
                # Visualize predictions
                vis.plot_distribution_grid(pred, ref, box_borders=box_borders, outdir=pred_dir,
                    start_ind=counter)
                vis.make_input_plots([X], outdir=pred_dir, start_ind=counter)

                counter += len(X)

    print(f'Done at rank {cfg["global_rank"]}. Total time: {time.perf_counter() - start_time:.0f}s')

    log_file.close()
    dist.barrier()
    dist.destroy_process_group()

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
    mp.set_start_method('spawn')
    cfg['world_size'] = int(os.environ['WORLD_SIZE'])
    cfg['global_rank'] = int(os.environ['RANK'])
    cfg['local_rank'] = int(os.environ['LOCAL_RANK'])
    run(cfg)
