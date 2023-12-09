# Geometric Data Analysis
# November 2023
#
# Ben Kabongo
# M2 MVA, ENS Paris-Saclay


import enum
import numpy as np
import scipy.optimize
from scipy.stats import rv_continuous
from typing import *

from sketching import sketching_operator as Sk


class FrequencyType(enum.Enum):
    GAUSSIAN = 0
    FOLDED_GAUSSIAN_RADIUS = 1
    ADAPTED_RADIUS = 2


class P_R(rv_continuous):
    def _pdf(self, R):
        return np.exp(-R**2 / 2) * (R**2 + R**4 / 4)**0.5


def draw_frequencies(X: np.ndarray, m: int, n0: int, m0: int, c: int, T: int, R: np.ndarray,
                    frequency_type: FrequencyType=FrequencyType.GAUSSIAN,
                    scipy_optimize_display: bool=False) -> float:

    n, d = X.shape

    assert n >= n0
    assert len(R) == c

    def _draw_frequencies(Sigma: np.ndarray, m: int):
        if frequency_type == FrequencyType.GAUSSIAN:
            return np.random.multivariate_normal(np.zeros(d), np.linalg.inv(Sigma), size=m)
        
        phi = np.random.normal(size=(m, d))
        phi /= np.repeat(np.linalg.norm(phi, axis=1), m).reshape((m, d))

        if frequency_type == FrequencyType.FOLDED_GAUSSIAN_RADIUS:
            R = np.random.normal(0, 1, size=m)
        elif frequency_type == FrequencyType.ADAPTED_RADIUS:
            R = P_R().rvs(size=m)
        else: raise ValueError("Unknown frequency type")
        
        Sigma_phi = phi @ np.linalg.inv(Sigma)
        return np.array([R[j] * Sigma_phi[j] for j in range(m)])
        

    sigma = np.array(1)
    for t in range(T):
        # Draw some frequencies adapted to the current σ ̄2
        Omega = _draw_frequencies(sigma * np.identity(d), m0)

        # Sort the frequencies {ω1, ..., ωm0 } by increasing radius ||ωj||2;
        w_norms = np.linalg.norm(Omega, axis=1)
        argsort = np.argsort(w_norms)[-1:0:-1]
        Omega = Omega[argsort]

        # Compute small empirical sketch
        sampled_X = X[np.random.choice(np.arange(n), size=n0, replace=False)]
        z0 = Sk(Omega, sampled_X) # weights = 1/n0

        # Divide sketch into blocks, find maximum peak in each block
        s = m0 // c
        e = np.zeros(c)
        for q in range(c):
            zj = z0[q * s : (q + 1) * s]
            e[q] = zj[np.argmax(np.linalg.norm(zj, axis=0))]
        
        # Update σ ̄2
        sigma0 = sigma
        sigma = scipy.optimize.minimize(
            fun=lambda sigma: _sigma_objective(sigma, e, R),
            constraints=({'type': 'ineq', 'fun': lambda x : x[0]}),
            jac=True,
            x0=sigma0,
            method="L-BFGS-B",
            options={"disp": scipy_optimize_display, "maxiter": 5}
        ).x

    Omega = _draw_frequencies(sigma * np.identity(d), m)
    return Omega


def _sigma_objective(sigma: np.ndarray, e: np.ndarray, R: np.ndarray) -> list[float, np.ndarray]:
    sigma = sigma[0]
    exp_Rq_sigma_div_2 = np.exp(-R * sigma/2)
    exp_Rq_sigma = np.exp(-R * sigma)
    f = (-2 * e * exp_Rq_sigma_div_2 + exp_Rq_sigma).sum()
    d_f = (R * (e * exp_Rq_sigma_div_2 - exp_Rq_sigma)).sum()
    return float(f), np.array([float(d_f)])


if __name__ == "__main__":
    n = 1_000_000
    n0 = 5_000
    d = 10
    m0 = 500
    m = 500
    c = 10
    R = np.random.random(c)
    T = 5
    Sigma = 0.5 * np.identity(d)
    X = np.random.multivariate_normal(np.zeros(d), Sigma, size=n)
    Omega = draw_frequencies(X, m, n0, m0, c, T, R, scipy_optimize_display=True)
    print(Omega.shape)
