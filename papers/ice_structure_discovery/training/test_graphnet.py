
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

import mlspm.data_loading as dl
import mlspm.preprocessing as pp
import mlspm.visualization as vis
from mlspm import graph, utils
from mlspm.models import GraphImgNet, PosNet
from mlspm.datasets import download_dataset


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
        peak_std                = cfg['peak_std'],
        device                  = device
    )
    model = GraphImgNet(
        n_classes               = len(cfg['classes']),
        posnet                  = posnet,
        iters                   = 5,
        node_feature_size       = 40,
        message_size            = 40,
        message_hidden_size     = 196,
        edge_cutoff             = cfg['edge_cutoff'],
        afm_cutoff              = cfg['afm_cutoff'],
        afm_res                 = cfg['box_res'][0],
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
    return model

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

    pp.rand_shift_xy_trend(X, max_layer_shift=0.02, max_total_shift=0.04)
    box_borders = graph.make_box_borders(X[0].shape[1:3], box_res[:2], z_lims)
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
    
    # Define model and load weights
    model = make_model(device, cfg)
    model.load_state_dict(torch.load(cfg['graphnet_weights']), strict=False)
    model.posnet.load_state_dict(torch.load(cfg['posnet_weights']))

    print(f'\n ========= Testing')

    eval_start = time.perf_counter()
    if cfg['timings']:
        t0 = eval_start

    stats = graph.GraphStats(cfg['classes'])
    _, test_loader = make_webDataloader(cfg, 'test')
    
    model.eval()
    with torch.no_grad():
        
        for ib, batch in enumerate(test_loader):
            
            # Transfer batch to device
            X, _, _, _, ref_graphs, _ = batch_to_device(batch, device)
            
            if cfg['timings']:
                torch.cuda.synchronize()
                t1 = time.perf_counter()

            # Gather statistical information
            pred_graphs, _ = model.predict_graph(X, bond_threshold=0.5)
            stats.add_batch(pred_graphs, ref_graphs)

            if (ib+1) % cfg['print_interval'] == 0:
                print(f'Test Batch {ib+1}')
            
            if cfg['timings']:
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                print(f'(Test) t0/Load Batch/Stats: {t1-t0:6f}/{t2-t1:6f}')
                t0 = t2

    # Save statistical information
    stats_dir = os.path.join(cfg['run_dir'], 'stats')
    stats.plot(stats_dir)
    stats.report(stats_dir)
    
    # Make predictions
    print(f'\n ========= Predict on {cfg["pred_batches"]} batches from the test set')
    counter = 0
    pred_dir = os.path.join(cfg['run_dir'], 'predictions/')
    
    with torch.no_grad():
        
        for ib, batch in enumerate(test_loader):
        
            if ib >= cfg['pred_batches']: break
            
            # Transfer batch to device
            X, _, _, _, ref_graphs, xyz = batch_to_device(batch, device)
            
            # Forward
            pred_graphs, grid_pred = model.predict_graph(X, bond_threshold=0.5)
            grid_pred = grid_pred.cpu().numpy()

            # Save xyzs
            utils.batch_write_xyzs(xyz, outdir=pred_dir, start_ind=counter)
            graph.save_graphs_to_xyzs(
                pred_graphs,
                cfg['classes'],
                outfile_format=os.path.join(pred_dir, '{ind}_graph_pred.xyz'),
                start_ind=counter
            )
            graph.save_graphs_to_xyzs(
                ref_graphs,
                cfg['classes'], 
                outfile_format=os.path.join(pred_dir, '{ind}_graph_ref.xyz'),
                start_ind=counter
            )
        
            # Visualize predictions
            box_borders = graph.make_box_borders(X.shape[1:3], cfg['box_res'][:2], cfg['z_lims'])
            grid_ref = graph.make_position_distribution(ref_graphs, box_borders, box_res=cfg['box_res'], std = cfg['peak_std'])
            vis.plot_graphs(
                pred_graphs,
                ref_graphs,
                box_borders=box_borders,
                outdir=pred_dir,
                start_ind=counter,
                classes=cfg['classes'],
                class_colors=cfg['class_colors']
            )
            vis.plot_distribution_grid(
                grid_pred,
                grid_ref,
                box_borders=box_borders,
                outdir=pred_dir,
                start_ind=counter
            )

            # Plot input AFM images
            X = X.cpu().numpy()
            vis.make_input_plots([X], outdir=pred_dir, start_ind=counter)

            counter += len(X)

    print(f'Done. Total time: {time.perf_counter() - start_time:.0f}s')

if __name__ == '__main__':

    graphnet_fit_dir = Path('./fit_graphnet_Cu111')
    posnet_fit_dir = Path('./fit_posnet_Cu111')

    # Get config
    config_path = Path(graphnet_fit_dir) / 'config.yaml'
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    cfg['run_dir'] = './test_graphnet'
    cfg['posnet_weights'] = str(posnet_fit_dir / 'best_model.pth')
    cfg['graphnet_weights'] = str(graphnet_fit_dir / 'best_model.pth')
    cfg['world_size'] = 1

    run_dir = Path(cfg['run_dir'])
    run_dir.mkdir(exist_ok=True, parents=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        # Remember settings
        yaml.safe_dump(cfg, f)

    # Set random seeds
    torch.manual_seed(cfg['random_seed'])
    random.seed(cfg['random_seed'])
    np.random.seed(cfg['random_seed'])

    # Download the dataset if it's not already there
    download_dataset(cfg['dataset'], cfg['data_dir'])

    # Start test
    run(cfg)
