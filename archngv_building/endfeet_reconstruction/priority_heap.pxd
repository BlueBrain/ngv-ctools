cimport numpy as np


ctypedef np.npy_int64 index_t
ctypedef np.npy_float32 float_t

cdef struct PriorityHeapRecord

cdef class MinPriorityHeap:
    cdef index_t capacity
    cdef index_t heap_ptr
    cdef PriorityHeapRecord* heap_
    cdef index_t* idmap_

    cdef bint is_empty(self) nogil
    cdef index_t push(self, index_t node_id, float_t value) nogil except -1
    cdef index_t pop(self, index_t* node_id, float_t* value) nogil
    cdef void heapify_up(self, PriorityHeapRecord* heap, index_t* idmap, index_t pos) nogil
    cdef void heapify_down(self, PriorityHeapRecord* heap, index_t* idmap, index_t pos, index_t heap_length) nogil

    # for testing
    cpdef index_t py_push(self, index_t node_id, float value)
    cpdef tuple py_pop(self)
