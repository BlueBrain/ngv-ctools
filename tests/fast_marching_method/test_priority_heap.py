from _ngv_ctools.fast_marching_method import MinPriorityHeap


def test_constructor():

    h = MinPriorityHeap(10)

    assert h.size == 0
    assert h.capacity == 10


def test_size_capacity():

    h = MinPriorityHeap(10)

    assert h.size == 0
    assert h.capacity == 10

    h.push(21, 0.1)
    h.push(10, 0.2)

    assert h.size == 2
    assert h.capacity == 10

    h.pop()

    assert h.size == 1
    assert h.capacity == 10

    h.pop()

    assert h.size == 0
    assert h.capacity == 10

    for i in range(11):
        h.push(i, 0.1 * i)

    assert h.size == 11
    assert h.capacity == 20


def test_single_value():

    h = MinPriorityHeap(100)

    h.push(11, 0.123)
    index, value = h.top()
    h.pop()

    assert index == 11
    assert value == 0.123


def test_increasing_values():

    h = MinPriorityHeap(100)

    h.push(0, 0.1)
    h.push(1, 0.2)

    index, value = h.top()
    assert index == 0
    assert value == 0.1

    h.pop()

    index, value = h.top()
    assert index == 1
    assert value == 0.2

    h.pop()

def test_decreasing_values():

    h = MinPriorityHeap(100)

    h.push(0, 0.2)
    h.push(1, 0.1)

    index, value = h.top()
    assert index == 1
    assert value == 0.1

    h.pop()

    index, value = h.top()
    assert index == 0
    assert value == 0.2

    h.pop()

def test_priority_heap():

    h = MinPriorityHeap(100)

    h.push(0, 0.4)
    h.push(1, 11.0)
    h.push(2, 1.2)
    h.push(3, 100.0)
    h.push(4, 51.0)
    h.push(5, 100000.0)

    expected_ids = [0, 2, 1, 4, 3, 5]
    expected_values = [0.4, 1.2, 11.0, 51.0, 100.0, 100000.0]

    for expected_id, expected_value in zip(expected_ids, expected_values):

        node_id, value = h.top()
        h.pop()
        assert expected_id == node_id, (node_id, value)
        assert expected_value == value, (node_id, value)
