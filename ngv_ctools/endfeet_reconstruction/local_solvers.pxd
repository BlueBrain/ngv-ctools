
cdef float local_solver_2D(float Ax, float Ay, float Az,
                            float Bx, float By, float Bz,
                            float Cx, float Cy, float Cz,
                            float TA, float TB) nogil

cpdef float py_local_solver_2D(float[:] A, float[:] B, float[:] C, float TA, float TB)

cpdef tuple second_order_solutions(float a, float b, float c)
