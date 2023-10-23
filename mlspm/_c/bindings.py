
import os
import numpy as np
import multiprocessing as mp

import ctypes
from ctypes import c_int, c_float, POINTER

N_POOL_PROC = mp.cpu_count()

this_dir = os.path.dirname(os.path.abspath(__file__))
clib = ctypes.CDLL(os.path.join(this_dir, 'asdafm_lib.so'))

fp_p = POINTER(c_float)
int_p = POINTER(c_int)
clib.match_template_mad.argtypes = [
    c_int, c_int, c_int, fp_p,
    c_int, c_int, c_int, fp_p,
    fp_p
]
clib.match_template_msd.argtypes = clib.match_template_mad.argtypes
clib.match_template_mad_norm.argtypes = clib.match_template_mad.argtypes
clib.match_template_msd_norm.argtypes = clib.match_template_mad.argtypes
clib.match_template_mad_2d.argtypes = [
    c_int, c_int, fp_p,
    c_int, c_int, fp_p,
    fp_p
]
clib.match_template_msd_2d.argtypes = clib.match_template_mad_2d.argtypes
clib.match_template_mad_norm_2d.argtypes = clib.match_template_mad_2d.argtypes
clib.match_template_msd_norm_2d.argtypes = clib.match_template_mad_2d.argtypes

clib.peak_dist.argtypes = [
    c_int, c_int, c_int, c_int, fp_p,
    int_p, fp_p,
    fp_p, fp_p, c_float
]

def match_template(array, template, method='mad'):

    nax, nay, naz = array.shape
    ntx, nty, ntz = template.shape
    array_c = array.astype(np.float32).ctypes.data_as(fp_p)
    template_c = template.astype(np.float32).ctypes.data_as(fp_p)
    dist_array = np.empty((nax, nay, naz), dtype=np.float32)
    dist_array_c = dist_array.ctypes.data_as(fp_p)

    if method == 'mad':
        clib.match_template_mad(
            nax, nay, naz, array_c,
            ntx, nty, ntz, template_c,
            dist_array_c
        )
    elif method == 'msd':
        clib.match_template_msd(
            nax, nay, naz, array_c,
            ntx, nty, ntz, template_c,
            dist_array_c
        )
    elif method == 'mad_norm':
        clib.match_template_mad_norm(
            nax, nay, naz, array_c,
            ntx, nty, ntz, template_c,
            dist_array_c
        )
    elif method == 'msd_norm':
        clib.match_template_msd_norm(
            nax, nay, naz, array_c,
            ntx, nty, ntz, template_c,
            dist_array_c
        )
    else:
        raise ValueError(f'Unknown matching method `{method}`.')
        
    return dist_array

def match_template_pool(arrays, template, method='mad'):
    inp = [(array, template, method) for array in arrays]
    with mp.Pool(processes=N_POOL_PROC) as pool:
        dist_arrays = pool.starmap(match_template, inp)
    dist_arrays = np.stack(dist_arrays, axis=0)
    return dist_arrays

def match_template_2d(array, template, method='mad'):

    nax, nay = array.shape
    ntx, nty = template.shape
    array_c = array.astype(np.float32).ctypes.data_as(fp_p)
    template_c = template.astype(np.float32).ctypes.data_as(fp_p)
    dist_array = np.empty((nax, nay), dtype=np.float32)
    dist_array_c = dist_array.ctypes.data_as(fp_p)

    if method == 'mad':
        clib.match_template_mad_2d(
            nax, nay, array_c,
            ntx, nty, template_c,
            dist_array_c
        )
    elif method == 'msd':
        clib.match_template_msd_2d(
            nax, nay, array_c,
            ntx, nty, template_c,
            dist_array_c
        )
    elif method == 'mad_norm':
        clib.match_template_mad_norm_2d(
            nax, nay, array_c,
            ntx, nty, template_c,
            dist_array_c
        )
    elif method == 'msd_norm':
        clib.match_template_msd_norm_2d(
            nax, nay, array_c,
            ntx, nty, template_c,
            dist_array_c
        )
    else:
        raise ValueError(f'Unknown matching method `{method}`.')
        
    return dist_array

def peak_dist(atoms, n_xyz, xyz_start, xyz_step, std):

    nb = len(atoms)
    nx, ny, nz = n_xyz
    dist = np.empty([nb, nx, ny, nz], dtype=np.float32)
    dist_c = dist.ctypes.data_as(fp_p)
    N_atom = np.array([len(a) for a in atoms], dtype=np.int32).ctypes.data_as(int_p)
    pos = np.concatenate(atoms, axis=0).astype(np.float32).ctypes.data_as(fp_p)
    xyz_start = np.array(xyz_start, dtype=np.float32).ctypes.data_as(fp_p)
    xyz_step = np.array(xyz_step, dtype=np.float32).ctypes.data_as(fp_p)

    clib.peak_dist(
        nb, nx, ny, nz, dist_c,
        N_atom, pos,
        xyz_start, xyz_step, std)

    return dist