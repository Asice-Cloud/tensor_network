#!/usr/bin/env python3
"""
Fibonacci anyon demo (small):

 - Defines the Fibonacci F and R data
 - Builds the 2D fusion space for three τ anyons with total charge τ
 - Computes braid generators σ1 and σ2 matrices in that basis
 - Verifies braid relation and prints matrices/eigenvalues

Usage: python3 fibonacci_anyons.py
"""

#用F,R构造辨群,结合规则是 tau*tau = 1 + tau
#F把基从 (ττ)_a τ -> τ 变换到 τ (ττ)_a -> τ
#R是交换两个tau anyons时的相位

import numpy as np


def phi():
    return (1 + 5**0.5) / 2


def fibonacci_F():
    """Return the F-matrix for tau,tau,tau -> tau fusion in the basis { (ττ)_1 τ -> τ, (ττ)_τ τ -> τ }
    Standard convention (one common choice):
    F = [[φ^{-1}, φ^{-1/2}], [φ^{-1/2}, -φ^{-1}]]
    Base transformation matrix for changing fusion order.
    """
    ph = phi()
    inv = 1.0 / ph
    inv_sqrt = ph ** (-0.5)
    F = np.array([[inv, inv_sqrt], [inv_sqrt, -inv]], dtype=complex)
    return F


def fibonacci_R():
    """Return the R phases for tau x tau -> {1, tau} in the same ordering [1, tau].
    Commonly used values:
      R^{1}_{ττ} = exp(-4π i / 5)
      R^{τ}_{ττ} = exp( 3π i / 5)
    See literature on Fibonacci anyons / SU(2)_3.
    Exchange phases for two τ anyons.
    """
    R1 = np.exp(-4j * np.pi / 5)
    Rtau = np.exp(3j * np.pi / 5)
    return np.array([R1, Rtau], dtype=complex)


def braid_matrices_3tau():
    """Construct σ1 and σ2 for 3 τ anyons with total charge τ in the basis where
    the first fusion is (τ τ) -> a with a in {1, τ} (and then (a τ) -> τ).
    - σ1 acts on first two anyons: diagonal in this basis with entries R_a
    - σ2 acts on second/third anyons: need to change basis with F, then apply diag(R), then F^{-1}
    Returns (sigma1, sigma2)
    """
    F = fibonacci_F()
    R = fibonacci_R()
    # basis order a=1, a=tau
    sigma1 = np.diag(R)
    # sigma2 = F^{-1} diag(R) F
    Finv = np.linalg.inv(F)
    sigma2 = Finv @ np.diag(R) @ F
    return sigma1, sigma2, F, R


def main():
    sigma1, sigma2, F, R = braid_matrices_3tau()

    np.set_printoptions(precision=4, suppress=True)
    print('Fibonacci constants: φ =', phi())
    print('\nF matrix (τττ→τ basis):\n', F)
    print('\nR phases (for channels [1, τ]):', R)
    print('\nσ1 (acts on first pair) = diag(R):\n', sigma1)
    print('\nσ2 (acts on second pair) = F^{-1} diag(R) F:\n', sigma2)

    # check braid relation σ1 σ2 σ1 = σ2 σ1 σ2
    lhs = sigma1 @ sigma2 @ sigma1
    rhs = sigma2 @ sigma1 @ sigma2
    diff_norm = np.linalg.norm(lhs - rhs)
    print('\n||σ1σ2σ1 - σ2σ1σ2|| =', diff_norm)

    # commutator
    comm = sigma1 @ sigma2 - sigma2 @ sigma1
    print('\nCommutator [σ1,σ2] =\n', comm)

    # eigenvalues
    ev1, _ = np.linalg.eig(sigma1)
    ev2, _ = np.linalg.eig(sigma2)
    print('\nEigenvalues σ1:', ev1)
    print('Eigenvalues σ2:', ev2)

    # show that these matrices are non-commuting unitaries
    print('\nCheck unitarity: σ1†σ1 - I norm =', np.linalg.norm(sigma1.conj().T @ sigma1 - np.eye(2)))
    print('Check unitarity: σ2†σ2 - I norm =', np.linalg.norm(sigma2.conj().T @ sigma2 - np.eye(2)))


if __name__ == '__main__':
    main()
