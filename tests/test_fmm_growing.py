import os
from pathlib import Path

import numpy as np
from numpy import testing as npt
import openmesh

from ngv_ctools.endfeet_reconstruction.fmm_growing import solve


DATA_DIRECTORY = Path(__file__).parent / Path('data')

def _print_plane(row_size, expected_groups, groups):

    class Colors:

        RED = '\u001b[31m'
        CYAN = '\u001b[36m'
        GREEN = '\u001b[32m'
        YELLOW = '\u001b[33m'
        END  = '\033[0m'

    def row2str(row):
        return ' '.join([f'{colors[group]}{group:>2}{Colors.END}' for group in row])

    colors = {0: Colors.RED, 1: Colors.CYAN, 2:Colors.GREEN, 3: Colors.YELLOW,  -1: ''}

    matrix1 = expected_groups.reshape(-1, row_size)
    matrix2 = groups.reshape(-1, row_size)

    txt = ' ' * 5 + 'Expected' + ' ' * 5 + '\t' + ' ' * 6 + 'Result\n'

    for row1, row2 in zip(matrix1, matrix2):
        txt += row2str(row1) + '\t' + row2str(row2) + '\n'

    print('Expected:', repr(expected_groups))
    print('Result  :', repr(groups))
    print(txt)

def _assign_vertex_neighbors(mesh):
    '''assign the neighbors for each vertex'''
    neighbors = mesh.vv_indices()
    mask = neighbors >= 0
    nn_offsets = np.count_nonzero(mask.reshape(neighbors.shape), axis=1)
    nn_offsets = np.hstack(((0, ), np.cumsum(nn_offsets))).astype(np.long)
    neighbors = neighbors[mask].astype(np.long)
    v_xyz = mesh.points().astype(np.float32)

    return neighbors, v_xyz, nn_offsets


def plane_10x10():

    filepath = os.path.join(DATA_DIRECTORY, 'plane_10x10.obj')

    mesh = openmesh.read_trimesh(filepath)
    neighbors, xyz, nn_offsets = _assign_vertex_neighbors(mesh)

    #n_vertices = 100
    #n_seeds = len(seeds)

    #v_status = np.full(n_vertices, fill_value=-1, dtype=np.long)
    #v_group_index = np.full(n_vertices, fill_value=-1, dtype=np.long)
    #v_travel_time = np.full(n_vertices, fill_value=np.inf, dtype=np.float32)

    #v_group_index[seeds] = np.arange(len(seeds))
    #v_status[seeds] = 1
    #v_travel_time[seeds] = 0.0

    return neighbors , nn_offsets, xyz




def test_solve__plane_10x10_two_seeds():

    seed_vertices = np.array([0, 99])
    neighbors, offsets, xyz = plane_10x10()

    v_group_index, v_travel_times, v_status = solve(neighbors, offsets, xyz, seed_vertices, 1000000.)

    expected_ids = np.array(
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
           0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
           0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
           0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
           0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
           0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
           0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
           0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
           0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    assert np.all(expected_ids == v_group_index), _print_plane(10, expected_ids, v_group_index)
    assert np.all(v_status == 1) # all are visited


def test_solve__plane_10x10_four_seeds():

    seed_vertices = np.array([0, 9, 90, 99])
    neighbors, offsets, xyz = plane_10x10()

    v_group_index, v_travel_times, v_status = solve(neighbors, offsets, xyz, seed_vertices, 1000000.)

    expected_ids = np.array(
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
         0, 0, 0, 2, 2, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 1, 1, 3, 3, 3,
         2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
         2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
         2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
         2, 2, 2, 2, 2, 3, 3, 3, 3, 3])

    assert np.all(expected_ids == v_group_index), _print_plane(10, expected_ids, v_group_index)
    assert np.all(v_status == 1) # all are visited
