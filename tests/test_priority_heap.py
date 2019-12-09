import pytest
from archngv_building.endfeet_reconstruction import priority_heap


def test_priority_heap():
    h = priority_heap.MinPriorityHeap(100)
    assert h.py_push(1, 1.) == 0
    assert h.py_pop() == (1, 1.)

    h.py_push(0, 0.4)
    h.py_push(1, 11.0)
    h.py_push(2, 1.2)
    h.py_push(3, 100.)
    h.py_push(4, 51.)
    h.py_push(5, 100000.)

    for expected_value, expected_id in sorted(((0.4, 0),
                                               (11.0, 1),
                                               (1.2, 2),
                                               (100., 3),
                                               (51., 4),
                                               (100000., 5),
                                               )):
        value, node_id = h.py_pop()
        assert pytest.approx(expected_id) == node_id
        assert pytest.approx(expected_value) == value
