# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
from libc.math cimport fabs, sqrt, fmin


DEF EPS = 1e-6


cdef inline float _dot(float x0, float x1, float x2, float y0, float y1, float y2) nogil:
    return x0 * y0 + x1 * y1 + x2 * y2


cdef inline float _euclidean_norm2(float x, float y, float z) nogil:
    return _dot(x, y, z, x, y, z)


cdef void _second_order_solutions(float a, float b, float c,
                                  float* root1, float* root2) nogil:

    root1[0] = root2[0] = -1.

    # numerical inaccuracy
    if fabs(b) < EPS:
        b = 0.

    #  first order equation if a is zero
    if fabs(a) < EPS:
        if b != 0.:
            root1[0] = -c / b
        return

    cdef float delta = b ** 2 - 4. * a * c  # discriminant

    # no real roots, nothing to do here
    if delta < 0.:
        return

    if delta < EPS:
        root1[0] = -b / (2. * a)
    elif delta > 0.:
        # Numerically Stable Method for Solving Quadratic Equations
        # https://people.csail.mit.edu/bkph/articles/Quadratics.pdf
        if b >= 0.:
            root1[0] = 0.5 * (-b - sqrt(delta)) / a
            root2[0] = 2. * c / (-b - sqrt(delta))
        else:
            root1[0] = 0.5 * (-b + sqrt(delta)) / a
            root2[0] = 2. * c / (-b + sqrt(delta))


cdef float local_solver_2D(float Ax, float Ay, float Az,
                           float Bx, float By, float Bz,
                           float Cx, float Cy, float Cz,
                           float TA, float TB) nogil:

    """solve for travel_time in triangle when two vertice's times and the geometry are known

    Following: A FAST ITERATIVE METHOD FOR SOLVING THE EIKONAL EQUATION ON TRIANGULATED SURFACES
    doi: 10.1137/100788951

    Update the travel time at C taking into account
    the upwind neighbors A and B.

           C (v3)
          / \
         /   \
        /     \
       /       \
      /         \
      - - - - - -
    A (v1)       B (v2)

    TA is the travel time of the wavefront to A
    TB is the travel time of the wavefront to B
    """
    # AC vector
    cdef float ACx = Cx - Ax
    cdef float ACy = Cy - Ay
    cdef float ACz = Cz - Az

    # AB vector
    cdef float ABx = Bx - Ax
    cdef float ABy = By - Ay
    cdef float ABz = Bz - Az

    cdef float Caa = _dot(ACx, ACy, ACz, ACx, ACy, ACz)
    cdef float Cab = _dot(ACx, ACy, ACz, ABx, ABy, ABz)
    cdef float Cbb = _dot(ABx, ABy, ABz, ABx, ABy, ABz)

    cdef float A, B, C
    cdef float TAB = TB - TA

    cdef float l1 = -1, l2 = -1, l

    cdef bint l1_valid, l2_valid

    cdef float T31, T32

    if fabs(TAB) < EPS:
        l1 = Cab / Cbb
    else:
        inv_TAB_sq = 1. / TAB ** 2
        A = Cbb * (1. - Cbb * inv_TAB_sq)
        B = 2. * Cab * (-1. + Cbb * inv_TAB_sq)
        C = Caa - Cab * Cab * inv_TAB_sq
        _second_order_solutions(A, B, C, &l1, &l2)

    l1_valid = 0. <= l1 <= 1
    l2_valid = 0. <= l2 <= 1.

    if l1_valid and l2_valid:
        # solutions can be symmetric. In that case pick the one that gives the shortest
        # travel time (Fermat's principle).
        T31 = TA + l1 * TAB + sqrt(_euclidean_norm2(ACx - l1 * ABx, ACy - l1 * ABy, ACz - l1 * ABz))
        T32 = TA + l2 * TAB + sqrt(_euclidean_norm2(ACx - l2 * ABx, ACy - l2 * ABy, ACz - l2 * ABz))
        return fmin(T31, T32)
    elif not (l1_valid or l2_valid):
        # if no solution is found the characteristic of the gradient is outside the
        # triangle. In that case give the smallest travel time through the edges of
        # the triangle
        T31 = TA + sqrt(Caa)
        T32 = TB + sqrt(_euclidean_norm2(Cx - Bx, Cy - By, Cz - Bz))
        return fmin(T31, T32)
    else:
        l = l1 if l1_valid else l2
        return TA + l * TAB + sqrt(_euclidean_norm2(ACx - l * ABx, ACy - l * ABy, ACz - l * ABz))


cpdef tuple second_order_solutions(float a, float b, float c):
    '''for testing'''
    cdef float root1 = -1
    cdef float root2 = -1
    _second_order_solutions(a, b, c, &root1, &root2)
    return root1, root2


cpdef float py_local_solver_2D(float[:] A, float[:] B, float[:] C,
                               float TA, float TB):
    '''for testing'''
    return local_solver_2D(A[0], A[1], A[2],
                           B[0], B[1], B[2],
                           C[0], C[1], C[2],
                           TA,  TB)
