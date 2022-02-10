import os
import json
from pathlib import Path

import pytest
import numpy as np
from numpy import testing as npt

from _ngv_ctools.fast_marching_method import FastMarchingMethod
from ngv_ctools.fast_marching_method import grow_waves_on_triangulated_surface as solve

DATA_DIRECTORY = Path(__file__).parent / Path("data")


def _print_plane(row_size, expected_groups, groups):
    class Colors:

        RED = "\u001b[31m"
        CYAN = "\u001b[36m"
        GREEN = "\u001b[32m"
        YELLOW = "\u001b[33m"
        END = "\033[0m"

    def row2str(row):
        return " ".join([f"{colors[group]}{group:>2}{Colors.END}" for group in row])

    colors = {0: Colors.RED, 1: Colors.CYAN, 2: Colors.GREEN, 3: Colors.YELLOW, -1: ""}

    matrix1 = expected_groups.reshape(-1, row_size)
    matrix2 = groups.reshape(-1, row_size)

    txt = " " * 5 + "Expected" + " " * 5 + "\t" + " " * 6 + "Result\n"

    for row1, row2 in zip(matrix1, matrix2):
        txt += row2str(row1) + "\t" + row2str(row2) + "\n"

    print("Expected:", repr(expected_groups))
    print("Result  :", repr(groups))
    print(txt)


def _assign_vertex_neighbors(mesh):
    """assign the neighbors for each vertex"""
    neighbors = mesh.vv_indices()
    mask = neighbors >= 0
    nn_offsets = np.count_nonzero(mask.reshape(neighbors.shape), axis=1)
    nn_offsets = np.hstack(((0,), np.cumsum(nn_offsets))).astype(np.long)
    neighbors = neighbors[mask].astype(np.long)
    v_xyz = mesh.points().astype(np.float32)

    return neighbors, v_xyz, nn_offsets


def plane_10x10():

    # filepath = os.path.join(DATA_DIRECTORY, 'plane_10x10.obj')

    # mesh = openmesh.read_trimesh(filepath)
    # neighbors, xyz, nn_offsets = _assign_vertex_neighbors(mesh)

    neighbors = np.array([
        10,  1,  0, 10, 11,  2,  1, 11, 12,  3,  2, 12, 13,  4,  3, 13, 14,
        5,  4, 14, 15,  6,  5, 15, 16,  7,  6, 16, 17,  8,  7, 17, 18,  9,
        8, 18, 19, 20, 11,  1,  0, 21, 12,  2,  1, 10, 20, 22, 13,  3,  2,
       11, 21, 23, 14,  4,  3, 12, 22, 24, 15,  5,  4, 13, 23, 25, 16,  6,
        5, 14, 24, 26, 17,  7,  6, 15, 25, 27, 18,  8,  7, 16, 26, 28, 19,
        9,  8, 17, 27,  9, 18, 28, 29, 30, 21, 11, 10, 31, 22, 12, 11, 20,
       30, 32, 23, 13, 12, 21, 31, 33, 24, 14, 13, 22, 32, 34, 25, 15, 14,
       23, 33, 35, 26, 16, 15, 24, 34, 36, 27, 17, 16, 25, 35, 37, 28, 18,
       17, 26, 36, 38, 29, 19, 18, 27, 37, 19, 28, 38, 39, 40, 31, 21, 20,
       41, 32, 22, 21, 30, 40, 42, 33, 23, 22, 31, 41, 43, 34, 24, 23, 32,
       42, 44, 35, 25, 24, 33, 43, 45, 36, 26, 25, 34, 44, 46, 37, 27, 26,
       35, 45, 47, 38, 28, 27, 36, 46, 48, 39, 29, 28, 37, 47, 29, 38, 48,
       49, 50, 41, 31, 30, 51, 42, 32, 31, 40, 50, 52, 43, 33, 32, 41, 51,
       53, 44, 34, 33, 42, 52, 54, 45, 35, 34, 43, 53, 55, 46, 36, 35, 44,
       54, 56, 47, 37, 36, 45, 55, 57, 48, 38, 37, 46, 56, 58, 49, 39, 38,
       47, 57, 39, 48, 58, 59, 60, 51, 41, 40, 61, 52, 42, 41, 50, 60, 62,
       53, 43, 42, 51, 61, 63, 54, 44, 43, 52, 62, 64, 55, 45, 44, 53, 63,
       65, 56, 46, 45, 54, 64, 66, 57, 47, 46, 55, 65, 67, 58, 48, 47, 56,
       66, 68, 59, 49, 48, 57, 67, 49, 58, 68, 69, 70, 61, 51, 50, 71, 62,
       52, 51, 60, 70, 72, 63, 53, 52, 61, 71, 73, 64, 54, 53, 62, 72, 74,
       65, 55, 54, 63, 73, 75, 66, 56, 55, 64, 74, 76, 67, 57, 56, 65, 75,
       77, 68, 58, 57, 66, 76, 78, 69, 59, 58, 67, 77, 59, 68, 78, 79, 80,
       71, 61, 60, 81, 72, 62, 61, 70, 80, 82, 73, 63, 62, 71, 81, 83, 74,
       64, 63, 72, 82, 84, 75, 65, 64, 73, 83, 85, 76, 66, 65, 74, 84, 86,
       77, 67, 66, 75, 85, 87, 78, 68, 67, 76, 86, 88, 79, 69, 68, 77, 87,
       69, 78, 88, 89, 90, 81, 71, 70, 91, 82, 72, 71, 80, 90, 92, 83, 73,
       72, 81, 91, 93, 84, 74, 73, 82, 92, 94, 85, 75, 74, 83, 93, 95, 86,
       76, 75, 84, 94, 96, 87, 77, 76, 85, 95, 97, 88, 78, 77, 86, 96, 98,
       89, 79, 78, 87, 97, 79, 88, 98, 99, 91, 81, 80, 92, 82, 81, 90, 93,
       83, 82, 91, 94, 84, 83, 92, 95, 85, 84, 93, 96, 86, 85, 94, 97, 87,
       86, 95, 98, 88, 87, 96, 99, 89, 88, 97, 89, 98])

    xyz = np.array([
       [-1.      , -1.      , -0.      ],
       [-0.777778, -1.      , -0.      ],
       [-0.555556, -1.      , -0.      ],
       [-0.333333, -1.      , -0.      ],
       [-0.111111, -1.      , -0.      ],
       [ 0.111111, -1.      , -0.      ],
       [ 0.333333, -1.      , -0.      ],
       [ 0.555556, -1.      , -0.      ],
       [ 0.777778, -1.      , -0.      ],
       [ 1.      , -1.      , -0.      ],
       [-1.      , -0.777778, -0.      ],
       [-0.777778, -0.777778, -0.      ],
       [-0.555556, -0.777778, -0.      ],
       [-0.333333, -0.777778, -0.      ],
       [-0.111111, -0.777778, -0.      ],
       [ 0.111111, -0.777778, -0.      ],
       [ 0.333333, -0.777778, -0.      ],
       [ 0.555556, -0.777778, -0.      ],
       [ 0.777778, -0.777778, -0.      ],
       [ 1.      , -0.777778, -0.      ],
       [-1.      , -0.555556, -0.      ],
       [-0.777778, -0.555556, -0.      ],
       [-0.555556, -0.555556, -0.      ],
       [-0.333333, -0.555556, -0.      ],
       [-0.111111, -0.555556, -0.      ],
       [ 0.111111, -0.555556, -0.      ],
       [ 0.333333, -0.555556, -0.      ],
       [ 0.555556, -0.555556, -0.      ],
       [ 0.777778, -0.555556, -0.      ],
       [ 1.      , -0.555556, -0.      ],
       [-1.      , -0.333333, -0.      ],
       [-0.777778, -0.333333, -0.      ],
       [-0.555556, -0.333333, -0.      ],
       [-0.333333, -0.333333, -0.      ],
       [-0.111111, -0.333333, -0.      ],
       [ 0.111111, -0.333333, -0.      ],
       [ 0.333333, -0.333333, -0.      ],
       [ 0.555556, -0.333333, -0.      ],
       [ 0.777778, -0.333333, -0.      ],
       [ 1.      , -0.333333, -0.      ],
       [-1.      , -0.111111, -0.      ],
       [-0.777778, -0.111111, -0.      ],
       [-0.555556,-0.111111, -0.      ],
       [-0.333333, -0.111111, -0.      ],
       [-0.111111, -0.111111, -0.      ],
       [ 0.111111, -0.111111, -0.      ],
       [ 0.333333, -0.111111, -0.      ],
       [ 0.555556, -0.111111, -0.      ],
       [ 0.777778, -0.111111, -0.      ],
       [ 1.      , -0.111111, -0.      ],
       [-1.      ,  0.111111,  0.      ],
       [-0.777778,  0.111111,  0.      ],
       [-0.555556,  0.111111,  0.      ],
       [-0.333333,  0.111111,  0.      ],
       [-0.111111,  0.111111,  0.      ],
       [ 0.111111,  0.111111,  0.      ],
       [ 0.333333,  0.111111,  0.      ],
       [ 0.555556,  0.111111,  0.      ],
       [ 0.777778,  0.111111,  0.      ],
       [ 1.      ,  0.111111,  0.      ],
       [-1.      ,  0.333333,  0.      ],
       [-0.777778,  0.333333,  0.      ],
       [-0.555556,  0.333333,  0.      ],
       [-0.333333,  0.333333,  0.      ],
       [-0.111111,  0.333333,  0.      ],
       [ 0.111111,  0.333333,  0.      ],
       [ 0.333333,  0.333333,  0.      ],
       [ 0.555556,  0.333333,  0.      ],
       [ 0.777778,  0.333333,  0.      ],
       [ 1.      ,  0.333333,  0.      ],
       [-1.      ,  0.555556,  0.      ],
       [-0.777778,  0.555556,  0.      ],
       [-0.555556,  0.555556,  0.      ],
       [-0.333333,  0.555556,  0.      ],
       [-0.111111,  0.555556,  0.      ],
       [ 0.111111,  0.555556,  0.      ],
       [ 0.333333,  0.555556,  0.      ],
       [ 0.555556,  0.555556,  0.      ],
       [ 0.777778,  0.555556,  0.      ],
       [ 1.      ,  0.555556,  0.      ],
       [-1.      ,  0.777778,  0.      ],
       [-0.777778,  0.777778,  0.      ],
       [-0.555556,  0.777778,  0.      ],
       [-0.333333,  0.777778,  0.      ],
       [-0.111111,  0.777778,  0.      ],
       [ 0.111111,  0.777778,  0.      ],
       [ 0.333333,  0.777778,  0.      ],
       [ 0.555556,  0.777778,  0.      ],
       [ 0.777778,  0.777778,  0.      ],
       [ 1.      ,  0.777778,  0.      ],
       [-1.      ,  1.      ,  0.      ],
       [-0.777778,  1.      ,  0.      ],
       [-0.555556,  1.      ,  0.      ],
       [-0.333333,  1.      ,  0.      ],
       [-0.111111,  1.      ,  0.      ],
       [ 0.111111,  1.      ,  0.      ],
       [ 0.333333,  1.      ,  0.      ],
       [ 0.555556,  1.      ,  0.      ],
       [ 0.777778,  1.      ,  0.      ],
       [ 1.      ,  1.      ,  0.      ]], dtype=np.float32)

    nn_offsets = np.array([
        0,   2,   6,  10,  14,  18,  22,  26,  30,  34,  37,  41,  47,
        53,  59,  65,  71,  77,  83,  89,  93,  97, 103, 109, 115, 121,
       127, 133, 139, 145, 149, 153, 159, 165, 171, 177, 183, 189, 195,
       201, 205, 209, 215, 221, 227, 233, 239, 245, 251, 257, 261, 265,
       271, 277, 283, 289, 295, 301, 307, 313, 317, 321, 327, 333, 339,
       345, 351, 357, 363, 369, 373, 377, 383, 389, 395, 401, 407, 413,
       419, 425, 429, 433, 439, 445, 451, 457, 463, 469, 475, 481, 485,
       488, 492, 496, 500, 504, 508, 512, 516, 520, 522])

    # n_vertices = 100
    # n_seeds = len(seeds)

    # v_status = np.full(n_vertices, fill_value=-1, dtype=np.long)
    # v_group_index = np.full(n_vertices, fill_value=-1, dtype=np.long)
    # v_travel_time = np.full(n_vertices, fill_value=np.inf, dtype=np.float32)

    # v_group_index[seeds] = np.arange(len(seeds))
    # v_status[seeds] = 1
    # v_travel_time[seeds] = 0.0

    return neighbors, nn_offsets, xyz


def test_fmm_constructor():

    neighbors, offsets, xyz = plane_10x10()

    seed_vertices = np.array([0])

    fmm = FastMarchingMethod(neighbors, offsets, xyz, seed_vertices, 100000.)

    npt.assert_array_equal(fmm.seeds, seed_vertices)

    for i, vertex in enumerate(fmm.vertices):
        assert vertex.group_id == -1
        assert np.isinf(vertex.travel_time)
        npt.assert_array_equal(vertex.neighbors, neighbors[offsets[i]: offsets[i + 1]])
        npt.assert_allclose(vertex.xyz, xyz[i])


@pytest.fixture
def ico_sphere_data():

    with open(DATA_DIRECTORY / "ico_sphere.json", "r") as fd:
        mesh_data = json.load(fd)

    vertices = np.array(mesh_data["vertices"], dtype=np.float32)
    neighbors = np.asarray(mesh_data["neighbors"], dtype=np.int64)
    offsets = np.asarray(mesh_data["nn_offsets"], dtype=np.int64)

    return vertices, neighbors, offsets


def test_ico_sphere__cutoff_distance(ico_sphere_data):
    """Test that the three seeds grown stop at the cutoff threshold
    """
    vertices, neighbors, offsets = ico_sphere_data

    squared_cutoff_distance = 0.25

    seed_vertices = np.array([0, 100, 250])
    group_indices, travel_times, statuses = solve(neighbors, offsets, vertices, seed_vertices, squared_cutoff_distance)

    for i, vertex in enumerate(seed_vertices):

        mask = group_indices == i
        assert mask.sum() > 0, "Seed did not grow."

        squared_distances = np.linalg.norm(vertices[mask] - vertices[vertex], axis=1) ** 2
        assert np.all(squared_distances < squared_cutoff_distance), (squared_distances, squared_cutoff_distance)


def _viz_growing(seed_vertices, group_indices, vertices, neighbors, offsets, expected_vertices):
    """Helper that uses vtk-viewer from bbpgitlab to viz the grown surfaces
    """
    from vtk_viewer import Viewer

    colormap =  {0: (255, 0, 0), 1: (0, 0, 255), 2: (0, 255, 0), -1: (255, 255, 255)}

    def create_triangles(group_indices):
        triangles = []
        triangle_colors = []
        for i in range(len(vertices)):
            ns = neighbors[offsets[i]: offsets[i + 1]]
            v0 = i
            for j in range(len(ns)):
                v1 = ns[j]
                v2 = ns[j + 1] if j < len(ns) - 1 else ns[0]
                triangles.append([v0, v1 ,v2])

                g1 = group_indices[v0]
                g2 = group_indices[v1]
                g3 = group_indices[v2]

                if (
                    (g1 == 0 and g2 == 0 and g3 == 0) or
                    (g1 == 0 and g2 == 0 and g3 != 0) or
                    (g1 != 0 and g2 == 0 and g3 == 0) or
                    (g1 == 0 and g2 != 0 and g3 == 0)
                    ):
                    triangle_colors.append(colormap[0])
                elif (
                    (g1 == 1 and g2 == 1 and g3 == 1) or
                    (g1 == 1 and g2 == 1 and g3 != 1) or
                    (g1 != 1 and g2 == 1 and g3 == 1) or
                    (g1 == 1 and g2 != 1 and g3 == 1)
                    ):
                    triangle_colors.append(colormap[1])
                elif (
                    (g1 == 2 and g2 == 2 and g3 == 2) or
                    (g1 == 2 and g2 == 2 and g3 != 2) or
                    (g1 != 2 and g2 == 2 and g3 == 2) or
                    (g1 == 2 and g2 != 2 and g3 == 2)
                    ):
                    triangle_colors.append(colormap[2])
                else:
                    triangle_colors.append(colormap[-1])

        triangles = np.array(triangles)
        triangle_colors = np.array(triangle_colors)

        return triangles, triangle_colors

    from vtk_viewer import Viewer
    v = Viewer()

    triangles, triangle_colors = create_triangles(group_indices)
    v.add_triangles(vertices, triangles, color=triangle_colors, wireframe=False)

    expected_group_indices = np.full_like(group_indices, fill_value=-1)

    for i, (seed, verts) in enumerate(expected_vertices.items()):
        expected_group_indices[verts] = i

    triangles, triangle_colors2 = create_triangles(expected_group_indices)

    is_diff = np.any(triangle_colors != triangle_colors2, axis=1)

    v.add_triangles(vertices, triangles[is_diff], color=triangle_colors2[is_diff], wireframe=True, line_width=2)

    v.render()


def test_ico_sphere__competition(ico_sphere_data):

    vertices, neighbors, offsets = ico_sphere_data

    seed_vertices = np.array([0, 100, 250])
    group_indices, travel_times, statuses = solve(neighbors, offsets, vertices, seed_vertices, 10000.)

    colormap = {0: (255, 0, 0), 1: (0, 0, 255), 2: (0, 255, 0), -1: (255, 255, 255)}

    expected_vertices = {
        0: [ 0,   3,   4,   5,  14,  15,  16,  17,  18,  19,  20,  21,  22,
            23,  24,  25,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,
            43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  59,  60,
            61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,
            74,  75,  76,  77,  78,  79,  80,  81,  89,  90,  91, 92, 115, 116,
           129, 130, 131, 132, 133, 142, 143, 144, 145, 337, 342, 343, 363,
           379, 380, 384, 385, 442, 447, 448, 449, 450, 451, 452, 463, 466,
           467, 468, 469, 470, 471, 472, 473, 520, 521, 523, 524, 525, 527, 528,
           529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541,
           542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554,
           555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567,
           568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580,
           581, 582, 583, 585, 586, 587, 589, 590, 591, 592, 594, 595, 596,
           597, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611,
           612, 613, 614, 615, 616, 617, 618, 619, 620, 624, 627, 628, 631,
           632, 633, 634, 636, 637, 638, 639, 640, 641],
        100: [ 1,   2,   6,   7,  12,  13,  26,  27,  28,  29,  30,  31,  32,
            54,  55,  56,  57,  58,  83,  84,  85,  86,  87,  88,  96,  97,
            98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
           111, 112, 113, 114, 117, 118, 119, 120, 121, 122, 123, 153, 154,
           155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,
           191, 192, 193, 206, 207, 284, 294, 295, 296, 297, 298, 299, 300,
           301, 302, 303, 304, 305, 312, 316, 317, 321, 322, 323, 324, 386,
           387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399,
           400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412,
           413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425,
           426, 427, 428, 429, 430, 431, 474, 475, 476, 477, 478, 479, 480,
           481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493,
           494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506,
           507, 508, 509, 510, 511, 512, 513, 514, 515, 517, 519, 522,
           526, 584, 588, 593, 598, 599, 621, 622, 623, 625, 626, 629, 630,
           635],
        250: [ 8,   9,  10,  11,  82, 93,  94,  95, 124, 125, 126, 127,
           128, 134, 135, 136, 137, 138, 139, 140, 141, 146, 147, 148, 149,
           150, 151, 152, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177,
           178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190,
           194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 208,
           209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221,
           222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234,
           235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247,
           248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260,
           261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273,
           274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 285, 286, 287,
           288, 289, 290, 291, 292, 293, 306, 307, 308, 309, 310, 311, 313,
           314, 315, 318, 319, 320, 325, 326, 327, 328, 329, 330, 331, 332,
           333, 334, 335, 336, 338, 339, 340, 341, 344, 345, 346, 347, 348,
           349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361,
           362, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375,
           376, 377, 378, 381, 382, 383, 432, 433, 434, 435, 436, 437, 438,
           439, 440, 441, 443, 444, 445, 446, 453, 454, 455, 456, 457, 458,
           459, 460, 461, 462, 464, 465, 516, 518]
    }

    # vizualize the results:
    # _viz_growing(seed_vertices, group_indices, vertices, neighbors, offsets, expected_vertices)

    for i, vertex in enumerate(seed_vertices):

        mask = group_indices == i
        assert mask.sum() > 0, "Seed did not grow."

        actual = np.where(mask)[0]
        expected = expected_vertices[vertex]
        print(i, vertex, np.setdiff1d(actual, expected), np.setdiff1d(expected, actual))
        npt.assert_array_equal(actual, expected)



def test_solve__plane_10x10_two_seeds__vertical():

    seed_vertices = np.array([0, 9])
    neighbors, offsets, xyz = plane_10x10()

    v_group_index, v_travel_times, v_status = solve(
        neighbors, offsets, xyz, seed_vertices, 1000000.0
    )

    expected_ids = np.array(
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    assert np.all(expected_ids == v_group_index), _print_plane(
        10, expected_ids, v_group_index
    )
    assert np.all(v_status == 1)  # all are visited


def test_solve__plane_10x10_two_seeds__horizontal():

    seed_vertices = np.array([5, 95])
    neighbors, offsets, xyz = plane_10x10()

    v_group_index, v_travel_times, v_status = solve(
        neighbors, offsets, xyz, seed_vertices, 1000000.0
    )

    expected_ids = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    assert np.all(expected_ids == v_group_index), _print_plane(
        10, expected_ids, v_group_index
    )
    assert np.all(v_status == 1)  # all are visited


def test_solve__plane_10x10_two_seeds__diagonal():

    seed_vertices = np.array([0, 99])
    neighbors, offsets, xyz = plane_10x10()

    v_group_index, v_travel_times, v_status = solve(
        neighbors, offsets, xyz, seed_vertices, 1000000.0
    )

    expected_ids = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
         0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
         0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
         0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
         0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    assert np.all(expected_ids == v_group_index), _print_plane(
        10, expected_ids, v_group_index
    )
    assert np.all(v_status == 1)  # all are visited


def test_solve__plane_10x10_four_seeds():

    seed_vertices = np.array([0, 9, 90, 99])
    neighbors, offsets, xyz = plane_10x10()

    v_group_index, v_travel_times, v_status = solve(
        neighbors, offsets, xyz, seed_vertices, 1000000.0
    )

    expected_ids = np.array(
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 1, 1, 1, 3,
         2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
         2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
         2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
         2, 2, 2, 2, 2, 3, 3, 3, 3, 3])

    assert np.all(expected_ids == v_group_index), _print_plane(
        10, expected_ids, v_group_index
    )
    assert np.all(v_status == 1)  # all are visited
