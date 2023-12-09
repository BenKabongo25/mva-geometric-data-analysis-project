# Geometric Data Analysis
# November 2023
#
# Ben Kabongo
# M2 MVA, ENS Paris-Saclay


import numpy as np


def sketching_operator(Omega: np.ndarray, Y: np.ndarray, beta: np.ndarray=None) -> np.ndarray:
    """
    :param Omega: m frequency vectors in R^n (m, n)
    :param Y: set of L points in R^n (L, n)
    :param beta: set of L weights in R (m,)
    :return the sketch of a set of L points Y with weights B in C^m (m,)
    """
    assert Y.shape[1] == Omega.shape[1]
    if beta is None: beta = np.ones(Y.shape[0]) / Y.shape[0]
    assert Y.shape[0] == beta.shape[0] 
    return np.array([(beta * np.exp(-(Y @ Omega[j])*1j)).sum() for j in range(Omega.shape[0])])


Sk = sketching_operator


if __name__ == "__main__":
    beta = np.random.random(10)
    beta /= beta.sum()
    Y = np.random.random((10, 5))
    Omega = np.random.random((3, 5))
    z = Sk(Omega, Y, beta)
    print(z)
    assert z.shape == (3,)
