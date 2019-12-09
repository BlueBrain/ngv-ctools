#cython: auto_pickle=False
#cython: boundscheck=False
cimport numpy as np

ctypedef np.npy_uintp SIZE_t


cdef list _subset_triangles_that_include_vertices(SIZE_t[:, :] triangles,
                                                   set indices):
    cdef SIZE_t n

    ret_indices = list()
    for n in range(len(triangles)):
        if(triangles[n, 0] in indices and
           triangles[n, 1] in indices and
           triangles[n, 2] in indices):
            ret_indices.append(n)

    return ret_indices


cpdef list subset_triangles_that_include_vertices(SIZE_t[:, :] triangles,
                                                   set indices):
    return _subset_triangles_that_include_vertices(triangles, indices)


cdef list _subset_triangles_that_do_not_include_vertices(SIZE_t[:, :] triangles,
                                                         set indices):
    cdef SIZE_t n
    ret_indices = list()
    for n in range(len(triangles)):
        if(triangles[n, 0] not in indices and
           triangles[n, 1] not in indices and
           triangles[n, 2] not in indices):
            ret_indices.append(n)

    return ret_indices

cpdef list subset_triangles_that_do_not_include_vertices(SIZE_t[:, :] triangles,
                                                         set indices):
    return _subset_triangles_that_do_not_include_vertices(triangles, indices)
