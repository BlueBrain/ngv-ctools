import numpy as np
import pytest
from numpy import testing as npt
from numpy.testing import assert_allclose

from _ngv_ctools.fast_marching_method import (
    dist_squared,
    dot,
    local_solver_2D,
    norm,
    second_order_solutions,
)


@pytest.mark.parametrize(
    "values",
    [
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        (1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        (1.0, 0.0, 1.0, 0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
    ],
)
def test_dot__zero_invariants(values):
    npt.assert_almost_equal(dot(*values), 0.0)


def test_dot():
    npt.assert_almost_equal(dot(1.0, 1.0, 1.0, 1.0, 1.0, 1.0), 3.0)

    npt.assert_almost_equal(
        dot(0.1, 0.2, 0.3, 1.0, 2.0, 3.0), 1.0 * 0.1 + 2.0 * 0.2 + 3.0 * 0.3
    )


@pytest.mark.parametrize("vector", [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
def test_norm__normalized(vector):
    npt.assert_almost_equal(norm(*vector), 1)


@pytest.mark.parametrize(
    "vector",
    [
        [-1.508, -9.972, 60.898],
        [57.632, 78.049, -64.655],
        [-75.318, -72.404, -5.292],
        [-25.212, -53.483, -17.493],
        [-26.627, 1.312, -69.258],
        [25.070, 74.667, 65.449],
        [-21.601, 14.817, 47.062],
        [51.800, 11.885, -3.473],
        [-82.055, -45.881, 70.474],
        [66.651, 52.015, 54.329],
    ],
)
def test_norm__vs_numpy(vector):
    npt.assert_almost_equal(norm(*vector), np.linalg.norm(vector), decimal=6)


def test_dist_squared():

    npt.assert_almost_equal(dist_squared([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]), 0.0)
    npt.assert_almost_equal(dist_squared([1.0, 1.0, 1.0], [2.0, 1.0, 1.0]), 1.0)
    npt.assert_almost_equal(dist_squared([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]), 3.0)

    npt.assert_allclose(
        dist_squared([-52.337, 84.592, 21.684], [70.481, -90.920, 29.878]), 45955.8646630,
    )

def _first_order_inputs():
    for b in range(-10, 10):
        if b != 0:
            for c in range(-10, 10):
                yield b, c, [-c / b, np.nan]

@pytest.mark.parametrize("b, c, expected_roots", _first_order_inputs())
def test_second_order_solutions__first_order(b, c, expected_roots):

    res = res = second_order_solutions(0.0, b, c)
    npt.assert_allclose(res, expected_roots)


def _second_order_random_inputs():

    np.random.seed(0)
    for _ in range(100):
        a, b, c = np.random.uniform(-1000., 1000., size=3)
        roots = np.roots([a, b, c])
        roots[~np.isreal(roots)] = np.nan
        yield a, b, c, roots


@pytest.mark.parametrize("a, b, c, expected_roots", _second_order_random_inputs())
def test_second_order_solutions__random(a, b, c, expected_roots):
    res = second_order_solutions(a, b, c)
    npt.assert_allclose(res, expected_roots)


def test_second_order_solutions():

    # all zero
    res = second_order_solutions(0.0, 0.0, 0.0)
    assert np.all(np.isnan(res))

    res = second_order_solutions(2.0, 0.0, 0.0)
    npt.assert_allclose(res, [0.0, np.nan])

    # simple roots
    r0, r1 = second_order_solutions(5.0, -4.0, -12.0)
    assert r0 == 2.0
    assert r1 == -1.2

    # q = discriminant is 0
    r0, r1 = second_order_solutions(3.0, -24.0, 48.0)
    assert r0 == 4.0
    assert np.isnan(r1)

    # b close to 0
    res = second_order_solutions(5.0, 0.000000001, -12.0)
    npt.assert_allclose(res, [-1.5491933822631836, 1.5491933822631836])

    # no solutions
    r0, r1 = second_order_solutions(1.0, -3.0, 4.0)
    assert np.isnan(r0)
    assert np.isnan(r1)


@pytest.mark.parametrize(
    "p",
    [
        (-1.0, 2.0, 3.0),
        (1.0, -2.0, -3.0),
        (5, -4.0, -12.0),
        (20.0, -15.0, -10.0),
        (1.0, 100.0, 200.0),
        (1000.0, 1.0, -5.0),
    ],
)
def test_second_order_solutions__roots(p):
    """Validate various coefficients using numpy's roots solver"""
    npt.assert_allclose(second_order_solutions(*p), np.roots(p))


def test_local_solver_2D__TAB_zero():
    ret = local_solver_2D(
        0., 0., 0.,
        0., 1., 0.,
        0.5, 0.5, 0.,
        1.0, 1.0
    )
    assert ret == 1.5

    a = np.array(
        [412.0400085449219, 1450.3499755859375, 228.718994140625]
    )
    b = np.array(
        [412.0950012207031, 1449.6800537109375, 233.41799926757812]
    )
    c = np.array(
        [413.510009765625, 1450.719970703125, 231.61000061035156]
    )
    ret = local_solver_2D(
        a[0],
        a[1],
        a[2],
        b[0],
        b[1],
        b[2],
        c[0],
        c[1],
        c[2],
        0.0,
        0.0
    )
    npt.assert_almost_equal(ret, 1.6326535940170288)


def _test_local_solver_2D_():

    a = np.array(
        [
            -6.390789985656738,
            1638.510009765625,
            681.5999755859375,
        ],
    )
    b = np.array(
        [
            -6.463850021362305,
            1637.7099609375,
            679.948974609375,
        ],
    )
    c = np.array(
        [
            -5.260650157928467,
            1636.300048828125,
            679.822998046875,
        ],
    )

    ret = local_solver_2D(
        a[0],
        a[1],
        a[2],
        b[0],
        b[1],
        b[2],
        c[0],
        c[1],
        c[2],
        45.13288497924805,
        46.620235443115234,
    )
    # oracle from commit 5522c94
    npt.assert_almost_equal(ret, 48.18464279174805)
