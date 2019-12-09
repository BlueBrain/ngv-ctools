# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False

import logging
cimport numpy as np

from .priority_heap cimport MinPriorityHeap, SIZE_t

from libc.math cimport isinf
from .local_solvers cimport local_solver_2D

L = logging.getLogger(__name__)


cpdef enum:
    FMM_FAR = -1
    FMM_TRIAL = 0
    FMM_KNOWN = 1

cpdef int UNKNOWN_GROUP = -1


cdef inline float _dist_squared(float[:, :] xyz,
                                SIZE_t ind1,
                                SIZE_t ind2) nogil:
    return ((xyz[ind1, 0] - xyz[ind2, 0]) ** 2 +
            (xyz[ind1, 1] - xyz[ind2, 1]) ** 2 +
            (xyz[ind1, 2] - xyz[ind2, 2]) ** 2)


cpdef float _find_travel_time(SIZE_t[:] nn_offsets,
                              float[:] v_travel_time,
                              SIZE_t[:] neighbors,
                              float[:, :] v_xyz,
                              SIZE_t ind) nogil:
    """ Update the vertex value by solving the eikonal
    equation using the first order discretization of the gradient
    """
    cdef:
        float TA, TB, TC, min_value = v_travel_time[ind]
        SIZE_t n, nb1, nb2
        SIZE_t nn_start = nn_offsets[ind]
        SIZE_t n_neighbors = nn_offsets[ind + 1] - nn_start

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


cdef void _update_neighbors(float squared_cutoff_distance,
                            SIZE_t[:] nn_offsets,
                            float[:] v_travel_time,
                            SIZE_t[:] neighbors,
                            float[:, :] v_xyz,
                            SIZE_t[:] v_status, #XXX char?
                            SIZE_t[:] group_labels,
                            long[:] v_group_index,
                            SIZE_t vertex_index,
                            MinPriorityHeap trial_heap) nogil:
    """ Update the values of the one ring neighbors of a vertex """
    cdef:
        SIZE_t n, nv, asdf
        SIZE_t nn_start = nn_offsets[vertex_index]
        SIZE_t n_neighbors = nn_offsets[vertex_index + 1] - nn_start

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


cpdef void solve(float squared_cutoff_distance,
                 SIZE_t[:] nn_offsets,
                 float[:] v_travel_time,
                 SIZE_t[:] neighbors,
                 float[:, :] v_xyz,
                 SIZE_t[:] v_status, #XXX char?
                 SIZE_t[:] group_labels,
                 long[:] v_group_index
                 ):
    """ Solves the eikonal equations using the fast marching method """

    cdef:
        MinPriorityHeap trial_heap
        float travel_time
        SIZE_t vertex_index
        SIZE_t n_vertices

    n_vertices = len(v_group_index)

    trial_heap = MinPriorityHeap(n_vertices)
    #L.info('Updating source neighbors...')
    # fist iterate over the known nodes and calculate the
    # travel time to the neighbors
    for vertex_index in range(n_vertices):
        if v_group_index[vertex_index] != UNKNOWN_GROUP:
            _update_neighbors(squared_cutoff_distance,
                              nn_offsets,
                              v_travel_time,
                              neighbors,
                              v_xyz,
                              v_status,
                              group_labels,
                              v_group_index,
                              vertex_index,
                              trial_heap)

    # expand in a breadth first manner from the smallest
    # distance node and update the travel times for the
    # propagation of the wavefront
    while not trial_heap.is_empty():
        # min travel time vertex
        trial_heap.pop(&vertex_index, &travel_time)

        if v_status[vertex_index] != FMM_KNOWN:
            v_status[vertex_index] = FMM_KNOWN
            v_travel_time[vertex_index] = travel_time

        _update_neighbors(squared_cutoff_distance,
                          nn_offsets,
                          v_travel_time,
                          neighbors,
                          v_xyz,
                          v_status,
                          group_labels,
                          v_group_index,
                          vertex_index,
                          trial_heap)
