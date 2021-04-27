from numpy.testing import assert_allclose
import numpy as np
from ngv_ctools.endfeet_reconstruction import local_solvers


def test_second_order_solutions():
    # simple roots
    r0, r1 = local_solvers.second_order_solutions(5, -4, -12)
    assert r0 == 2.
    assert r1 == -1.2000000476837158

    # q = discriminant is 0
    r0, r1 = local_solvers.second_order_solutions(3, -24, 48)
    assert r0 == 4.
    assert r1 == -1.

    # b close to 0
    r0, r1 = local_solvers.second_order_solutions(5, 0.000000001, -12)
    assert r0 == -1.5491933822631836
    assert r1 == 1.5491933822631836

    # no solutions
    r0, r1 = local_solvers.second_order_solutions(1, -3, 4)
    assert r0 == -1.
    assert r1 == -1.


def test_local_solver_2D_():
    a = np.array([0, 0, 0], dtype=np.float32)
    b = np.array([0, 1, 0], dtype=np.float32)
    c = np.array([0.5, 0.5, 0], dtype=np.float32)
    ret = local_solvers.py_local_solver_2D(a, b, c, 1., 1.)
    assert ret == 1.5

    a = np.array([-6.390789985656738, 1638.510009765625, 681.5999755859375, ], dtype=np.float32)
    b = np.array([-6.463850021362305, 1637.7099609375, 679.948974609375, ], dtype=np.float32)
    c = np.array([-5.260650157928467, 1636.300048828125, 679.822998046875, ], dtype=np.float32)

    ret = local_solvers.py_local_solver_2D(a, b, c, 45.13288497924805, 46.620235443115234)
    # oracle from commit 5522c94
    assert ret == 48.18464279174805

    a = np.array([412.0400085449219, 1450.3499755859375, 228.718994140625], dtype=np.float32)
    b = np.array([412.0950012207031, 1449.6800537109375, 233.41799926757812], dtype=np.float32)
    c = np.array([413.510009765625, 1450.719970703125, 231.61000061035156], dtype=np.float32)
    ret = local_solvers.py_local_solver_2D(a, b, c, 0., 0.)
    assert ret == 1.6326535940170288
