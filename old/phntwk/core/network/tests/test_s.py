import numpy as np
import skrf as rf
from siluxApi.phntwk.core.network.connection.connection import innerconnect_s
import timeit

sample_s = np.array([np.block([
    [
        np.ones((2, 2))*0.5, 
        np.zeros((2, 2))
    ],
    [
        np.zeros((2, 2)), 
        np.ones((2, 2))*0.6
    ]
]).astype(complex)])
for i in range(sample_s.shape[1]):
    sample_s[:,i, i] = 0

def test_vector_dot():
    sample_s = np.array([
        np.ones((2, 2))*0.5,
        np.ones((2, 2))*0.6,
    ], dtype=complex)
    print(sample_s[:, :, [0]] @ sample_s[:, [1], :])
    print(
        np.array( [
            np.ones((2, 2))*0.25,
            np.ones((2, 2))*0.36,
        ], dtype=complex)
    )
    assert np.all(
        sample_s[:, :, [0]] @ sample_s[:, [1], :] == 
        np.array( [
            np.ones((2, 2))*0.25,
            np.ones((2, 2))*0.36,
        ], dtype=complex)
    )

def test_s_block():
    s1 = np.ones((3, 2, 2))*0.5
    s2 = np.ones((3, 2, 2))*0.6
    s = np.block([
        [s1, np.zeros((3, 2, 2))],
        [np.zeros((3, 2, 2)), s2]
    ])
    return s

def test_interconnect_s():
    rf_s = rf.network.innerconnect_s(sample_s, 1, 2)
    my_s = innerconnect_s(sample_s, 1, 2)
    assert np.all(rf_s == my_s)

def test_rf_interconnect_s():
    return rf.network.innerconnect_s(sample_s, 1, 2)

def test_my_interconnect_s():
    return innerconnect_s(sample_s, 1, 2)


def main():
    test_s_block()
    # test_vector_dot()
    test_interconnect_s()
    # print(timeit.timeit(test_rf_interconnect_s, number=1000))
    # print(timeit.timeit(test_my_interconnect_s, number=1000))


if __name__ == "__main__":
    main()
    print()