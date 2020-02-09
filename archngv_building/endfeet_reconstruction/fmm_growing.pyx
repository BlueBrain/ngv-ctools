# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False

import logging
import numpy as np
cimport numpy as np

from .priority_heap cimport MinPriorityHeap, index_t, float_t

from libc.math cimport isinf
from .local_solvers cimport local_solver_2D

L = logging.getLogger(__name__)


cpdef enum:
    FMM_FAR = -1
    FMM_TRIAL = 0
    FMM_KNOWN = 1

cpdef index_t UNKNOWN_GROUP = -1


cdef inline float_t _dist_squared(float_t[:, :] xyz,
                                index_t ind1,
                                index_t ind2) nogil:
    return ((xyz[ind1, 0] - xyz[ind2, 0]) ** 2 +
            (xyz[ind1, 1] - xyz[ind2, 1]) ** 2 +
            (xyz[ind1, 2] - xyz[ind2, 2]) ** 2)


cpdef float_t _find_travel_time(index_t[:] nn_offsets,
                              float_t[:] v_travel_time,
                              index_t[:] neighbors,
                              float_t[:, :] v_xyz,
                              index_t ind) nogil:
    """ Update the vertex value by solving the eikonal
    equation using the first order discretization of the gradient
    """
    cdef:
        float_t TA, TB, TC, min_value = v_travel_time[ind]
        index_t n, nb1, nb2
        index_t nn_start = nn_offsets[ind]
        index_t n_neighbors = nn_offsets[ind + 1] - nn_start

    # find a triangle with nodes of known values and update the traveling time at vertex C
    for n in range(n_neighbors):

        if n == n_neighbors - 1:
            nb1 = neighbors[nn_start + n]
            # last edge cycled to the start
            nb2 = neighbors[nn_start]
        else:
            # consecutive pair
            nb1 = neighbors[nn_start + n]
            nb2 = neighbors[nn_start + n + 1]

        TA, TB = v_travel_time[nb1], v_travel_time[nb2]

        if isinf(TA) and isinf(TB):
            continue

        # ensure TB > TA with the ol good switcheroo
        if TB >= TA:
            TC = local_solver_2D(v_xyz[nb1, 0], v_xyz[nb1, 1], v_xyz[nb1, 2],
                                 v_xyz[nb2, 0], v_xyz[nb2, 1], v_xyz[nb2, 2],
                                 v_xyz[ind, 0], v_xyz[ind, 1], v_xyz[ind, 2],
                                 TA, TB)
        else:
            TC = local_solver_2D(v_xyz[nb2, 0], v_xyz[nb2, 1], v_xyz[nb2, 2],
                                 v_xyz[nb1, 0], v_xyz[nb1, 1], v_xyz[nb1, 2],
                                 v_xyz[ind, 0], v_xyz[ind, 1], v_xyz[ind, 2],
                                 TB, TA)

        min_value = min(min_value, TC)

    return min_value


cdef void _update_neighbors(float_t squared_cutoff_distance,
                            index_t[:] nn_offsets,
                            float_t[:] v_travel_time,
                            index_t[:] neighbors,
                            float_t[:, :] v_xyz,
                            index_t[:] v_status, #XXX char?
                            index_t[:] group_labels,
                            index_t[:] v_group_index,
                            index_t vertex_index,
                            MinPriorityHeap trial_heap) nogil:
    """ Update the values of the one ring neighbors of a vertex """
    cdef:
        index_t n, nv, asdf
        index_t nn_start = nn_offsets[vertex_index]
        index_t n_neighbors = nn_offsets[vertex_index + 1] - nn_start

    for n in range(n_neighbors):
        nv = neighbors[nn_start + n]

        # if the neighbor value has not been finalized (FMM_FAR, FMM_TRIAL)
        if v_status[nv] == FMM_KNOWN:
            continue

        # find the travel time of the wave to the neighbor vertex in the ring
        v_travel_time[nv] = _find_travel_time(nn_offsets,
                                              v_travel_time,
                                              neighbors,
                                              v_xyz,
                                              nv)

        # otherwise add in the priority queue with the travel time
        # as priority. It starts as a trial vertex.
        asdf = v_group_index[vertex_index]
        if(v_status[nv] == FMM_FAR and
           _dist_squared(v_xyz, nv, group_labels[asdf]) < squared_cutoff_distance):
            v_group_index[nv] = asdf
            v_status[nv] = FMM_TRIAL
            trial_heap.push(nv, v_travel_time[nv])


cpdef solve(index_t[:] neighbors,
            index_t[:] nn_offsets,
            float_t[:, :] v_xyz,
            index_t[:] seed_vertices,
            float_t squared_cutoff_distance):
    """
    Given a triangulation of N vertices, where the connectivity of the i-th vertex can be extracted
    via the neighbors and nn_offsets, and with coordinates v_xyz, solve the eikonal equation setting
    as starting points of the wave propagation the seed_vertices. If the wave propagation exceeds the
    squared_cutoff_distance then the spreading of that wave stops letting the neighboring seeds expand if any.

    Args:
        neighbors:
            Array of vertices stored so that the neighbors of the i-th vertex can be extracted
            neighbors[nn_offsets[i]: nn_offsets[i + 1]]

        nn_offsets:
            The offsets for extracting the neighbors for the i-th vertex

        v_xyz:
            The coordinates of the vertices

        seed_vertices:
            The vertex ids that will play the role of starting points in the simulation

        squared_cutoff_distance:
            The distance that the neighbor updating will stop for each vertex

    Returns:
        v_group_indices:
            The array of the indices to the seed_vertices or -1 the unassigned group

        v_travel_times:
            The travel times of the wavefronts for each vertex

        v_status:
            The fast marching method status, e.g. FMM_KNOWN, FMM_TRIAL, FMM_FAR for each
            vertex
    """
    cdef:
        float_t travel_time
        index_t i, vertex_index
        index_t n_seeds = len(seed_vertices)
        index_t n_vertices = len(v_xyz)
        index_t[:] v_status = np.full(n_vertices, fill_value=FMM_FAR, dtype=np.int64)
        float_t[:] v_travel_times = np.full(n_vertices, fill_value=np.inf, dtype=np.float32)
        index_t[:] v_group_indices = np.full(n_vertices, fill_value=UNKNOWN_GROUP, dtype=np.int64)

        MinPriorityHeap trial_heap = MinPriorityHeap(n_vertices)

    # initialize the seed vertices as already visited with
    # zero travel times and assign their respective groups
    for i in range(n_seeds):
        vertex_index = seed_vertices[i]
        v_status[vertex_index] = FMM_KNOWN
        v_travel_times[vertex_index] = 0.0
        v_group_indices[vertex_index] = i

    # update the 1-ring neighborhood of the seed vertices
    # and calculate the travel time to their neighbors
    for i in range(n_seeds):
        _update_neighbors(squared_cutoff_distance,
                          nn_offsets,
                          v_travel_times,
                          neighbors,
                          v_xyz,
                          v_status,
                          seed_vertices,
                          v_group_indices,
                          seed_vertices[i],
                          trial_heap)

    # expand in a breadth first manner from the smallest
    # distance node and update the travel times for the
    # propagation of the wavefront
    while not trial_heap.is_empty():

        # min travel time vertex
        trial_heap.pop(&vertex_index, &travel_time)

        if v_status[vertex_index] != FMM_KNOWN:
            v_status[vertex_index] = FMM_KNOWN
            v_travel_times[vertex_index] = travel_time

        _update_neighbors(squared_cutoff_distance,
                          nn_offsets,
                          v_travel_times,
                          neighbors,
                          v_xyz,
                          v_status,
                          seed_vertices,
                          v_group_indices,
                          vertex_index,
                          trial_heap)

    return np.asarray(v_group_indices), np.asarray(v_travel_times), np.asarray(v_status)
