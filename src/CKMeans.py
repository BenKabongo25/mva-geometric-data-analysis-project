# Geometric Data Analysis
# November 2023
#
# Ben Kabongo
# M2 MVA, ENS Paris-Saclay


import enum
import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.spatial.distance

from sketching import Sk


MAXITER = 5

class InitMode(enum.Enum):
    RANGE   = 0
    SAMPLE  = 1
    KPP     = 2



def CKMeans(z: np.ndarray, Omega: np.ndarray, K: int, l: np.ndarray, u: np.ndarray,
            X: np.ndarray=None, init: InitMode=InitMode.RANGE,
            scipy_optimize_display: bool=True) -> tuple:
    """CLOMPR for K-means (CKM) Algorithm
    :param z: Sketch in C^m (m,)
    :param Omega: frequencies in R^n (m, n)
    :param K: number of clusters
    :param l: lower bound
    :param u: upper bound
    :param X: data
    :param init: init mode
    :return C, alpha: centroids, weights
    """
    m, n = Omega.shape
    assert len(z) == m
    assert len(l) == len(u) == n

    r = z.copy()
    C = np.zeros((0, n))
    alpha = np.zeros(0)

    for t in range(1, 2 * K):
        # step 1 : Find a new centroid
        c0 = np.zeros(n)
        if init == InitMode.RANGE:
            c0 = np.array([np.random.uniform(li, ui) for li, ui in zip(l, u)])
        else:
            assert X is not None
            if init == InitMode.KPP and len(C) > 0:
                p = 1 / (scipy.spatial.distance.cdist(X, C).sum(axis=1) + 1e-6)
                p /= p.sum()
                idx = np.random.choice(np.arange(0, len(X)), p=p)
                c0 = X[idx]
            else:
                c0 = X[np.random.randint(0, len(X))]

        c = scipy.optimize.minimize(
            fun=lambda c: _step1_maximize_c(c, r, Omega, minimize=True),
            constraints=(scipy.optimize.LinearConstraint(np.identity(n), l, u),),
            x0=c0,
            jac=True,
            method="L-BFGS-B",
            options={"disp": scipy_optimize_display, "maxiter": MAXITER}
        ).x

        # Step 2 : Expand support
        C = np.vstack((C, c))
        current_K = len(C)

        # Step 3 : Enforce sparsity by Hard Thresholding if t > K
        if current_K > K:
            beta = scipy.optimize.minimize(
                fun=lambda beta: _step3_minimize_beta(beta, z, C, Omega),
                constraints=(scipy.optimize.LinearConstraint(np.identity(len(C)), 0),),
                jac=True,
                x0=np.ones(current_K) / current_K,
                method="L-BFGS-B",
                options={"disp": scipy_optimize_display, "maxiter": MAXITER}
            ).x

            # Select K largest entries β1 , ..., βK
            argsort = np.argsort(beta)[:K] 

            # Reduce the support 
            C = C[argsort]
            current_K = len(C)

        # Step 4 : Project to find alpha
        alpha0 = np.concatenate((alpha, [0] * (current_K - len(alpha))))
        alpha = scipy.optimize.minimize(
            fun=lambda alpha: _step4_minimize_alpha(alpha, z, C, Omega),
            constraints=(scipy.optimize.LinearConstraint(np.identity(current_K), 0, 1),),
            x0=alpha0,
            jac=True,
            method="L-BFGS-B",
            options={"disp": scipy_optimize_display, "maxiter": MAXITER}
        ).x

        # Step 5: Global gradient descent
        alpha_identity = scipy.linalg.block_diag(np.diag(np.zeros(current_K * n)), np.identity(current_K))
        C_identity = scipy.linalg.block_diag(np.identity(current_K * n), np.diag(np.zeros(current_K)))

        l_alpha = np.concatenate((np.repeat(l, current_K), [0] * current_K))
        u_alpha = np.concatenate((np.repeat(u, current_K), [+np.inf] * current_K))

        C_alpha0 = np.concatenate((C.flatten(), alpha))
        C_alpha = scipy.optimize.minimize(
            fun=lambda C_alpha: _step5_minimize_C_alpha(C_alpha, z, Omega, current_K),
            constraints=(
                scipy.optimize.LinearConstraint(alpha_identity, 0, 1),
                scipy.optimize.LinearConstraint(C_identity, l_alpha, u_alpha)
            ),
            x0=C_alpha0,
            jac=True,
            method="L-BFGS-B",
            options={"disp": scipy_optimize_display, "maxiter": MAXITER}
        ).x

        C, alpha = C_alpha[:-current_K].reshape((current_K, n)), C_alpha[-current_K:]

        # Update residual
        r = z - Sk(Omega, C, alpha)

    return C, alpha


def _step1_maximize_c(c: np.ndarray, r: np.ndarray, Omega: np.ndarray, minimize: bool=True) -> list[float, np.ndarray]:
    m = len(r)
    n = len(c)

    u1, d_u1 = 0, np.zeros(n)
    u2, d_u2 = 0, np.zeros(n)
    for k in range(m):
        w_k = Omega[k]
        theta_k = -w_k @ c
        u1 += (np.cos(theta_k)) ** 2 + (np.sin(theta_k)) ** 2
        u2 += 2 * np.cos(theta_k) * np.sin(theta_k)
        d_u1 += 4 * np.sin(theta_k) * np.cos(theta_k) * w_k
        d_u2 += 2 * (1 - 2 * np.cos(theta_k)) * w_k

    v = u1 ** 2 + u2 ** 2
    d_v = 2 * u1 * d_u1 + 2 * u2 * d_u2 

    u, d_u = 0, np.zeros(n)
    for j in range(m):
        r_j = r[j]
        a_j, b_j = np.real(r_j), np.imag(r_j)
        w_j = Omega[j]
        theta_j = -w_j @ c
        u1j = a_j * np.cos(theta_j) - b_j * np.sin(theta_j)
        u2j = a_j * np.sin(theta_j) + b_j * np.cos(theta_j)
        d_u1j = u2j * w_j
        d_u2j = -u1j * w_j
        u += u1j * u1 + u2j * u2
        d_u += d_u1j * u1 + u1j * d_u1 + d_u2j * u2 + u2j * d_u2

    d_c = (d_u * v + u * d_v) / (v ** 2)
    factor = -1 if minimize else +1
    return factor * u/v, factor * d_c


def _step3_minimize_beta(beta: np.ndarray, z: np.ndarray, C: np.ndarray, Omega: np.ndarray) -> list[float, np.ndarray]:
    K = len(beta)
    m = Omega.shape[0]
    Delta = np.zeros((K, m))
    for k in range(K):
        c_k = C[k][np.newaxis, :]
        delta_k = Sk(Omega, c_k)
        delta_k /= np.sqrt(delta_k @ delta_k)
        Delta[k] = delta_k
    d_beta = np.zeros_like(beta)
    for l in range(K):
        d_beta[l] = - 2 * beta[l] * z @ Delta[l] + np.sum([beta[k] * (Delta[k] @ Delta[l])])
    r = z - np.sum([beta[k] * Delta[k] for k in range(K)], axis=0)
    return np.real(r @ r), np.real(d_beta)


def _step4_minimize_alpha(alpha: np.ndarray, z: np.ndarray, C: np.ndarray, Omega: np.ndarray) -> list[float, np.ndarray]:
    d_alpha = np.zeros_like(alpha)
    K = len(d_alpha)
    m = len(z)

    for l in range(K):
        d_alpha_l = 0
        for j in range(m):
            z_j = z[j]
            a_j, b_j = np.real(z_j), np.imag(z_j)
            w_j = Omega[j]
            Theta_j = - C @ w_j
            alpha_cos_j = (alpha * np.cos(-Theta_j)).sum()
            alpha_sin_j = (alpha * np.sin(-Theta_j)).sum()
            theta_j_l = - w_j @ C[l]
            cos_j_l = np.cos(theta_j_l)
            sin_j_l = np.sin(theta_j_l)
            d_alpha_l += -a_j * cos_j_l + b_j * sin_j_l + 2 * cos_j_l * alpha_cos_j - 2 * sin_j_l * alpha_sin_j
        d_alpha[l] = d_alpha_l

    r = z - Sk(Omega, C, alpha)
    return np.real(r @ r), d_alpha


def _step5_minimize_C_alpha(C_alpha: np.ndarray, z: np.ndarray, Omega: np.ndarray, K: int) -> list[float, np.ndarray]:
    alpha = C_alpha[-K:]
    C = C_alpha[:-K].reshape((K, -1))
    
    m = len(z)
    n = C.shape[1]
    d_C = np.zeros((K, n))
    d_alpha = np.zeros(K)

    for l in range(K):
        d_alpha_l = 0
        d_C_l = np.zeros(n)
        for j in range(m):
            z_j = z[j]
            a_j, b_j = np.real(z_j), np.imag(z_j)
            w_j = Omega[j]
            Theta_j = - C @ w_j
            alpha_cos_j = (alpha * np.cos(-Theta_j)).sum()
            alpha_sin_j = (alpha * np.sin(-Theta_j)).sum()
            theta_j_l = - w_j @ C[l]
            cos_j_l = np.cos(theta_j_l)
            sin_j_l = np.sin(theta_j_l)
            d_alpha_l += -a_j * cos_j_l + b_j * sin_j_l + 2 * cos_j_l * alpha_cos_j - 2 * sin_j_l * alpha_sin_j
            d_C_l += (a_j * sin_j_l + b_j * cos_j_l + 2 * sin_j_l * alpha_cos_j + 2 * cos_j_l * alpha_sin_j) * alpha[l] * w_j
        d_alpha[l] = d_alpha_l
        d_C[l] = d_C_l

    d_C_alpha = np.zeros_like(C_alpha)
    d_C_alpha[:-K] = d_C.flatten()
    d_C_alpha[-K:] = d_alpha

    r = z - Sk(Omega, C, alpha)
    return np.real(r @ r), d_C_alpha

