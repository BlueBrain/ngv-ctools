import numpy as np
from archngv_building.endfeet_reconstruction import endfoot


def test_subset_triangles_that_include_vertices():
    vertices = np.array([[1, 2, 3],
                         [1, 2, 4],
                         ], dtype=np.uint64)
    indices = {1, 2, 3}
    ret = endfoot.subset_triangles_that_include_vertices(vertices, indices)
    assert ret == [0]

def test_subset_triangles_that_do_not_include_vertices():
    vertices = np.array([[1, 2, 3],
                         [1, 2, 4],
                         [5, 6, 7],
                         ], dtype=np.uint64)
    indices = {1, 2, 3}
    ret = endfoot.subset_triangles_that_do_not_include_vertices(vertices, indices)
    assert ret == [2]
