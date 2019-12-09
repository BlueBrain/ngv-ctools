cimport numpy as np

ctypedef np.npy_long SIZE_t

cdef struct PriorityHeapRecord

cdef class MinPriorityHeap:
    cdef SIZE_t capacity
    cdef SIZE_t heap_ptr
    cdef PriorityHeapRecord* heap_
    cdef SIZE_t* idmap_

    cdef bint is_empty(self) nogil
    cdef int push(self, SIZE_t node_id, float value) nogil except -1
    cdef int pop(self, SIZE_t* node_id, float* value) nogil
    cdef void heapify_up(self, PriorityHeapRecord* heap, SIZE_t* idmap, SIZE_t pos) nogil
    cdef void heapify_down(self, PriorityHeapRecord* heap, SIZE_t* idmap, SIZE_t pos, SIZE_t heap_length) nogil

    # for testing
    cpdef int py_push(self, SIZE_t node_id, float value)
    cpdef tuple py_pop(self)
