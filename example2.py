import numpy as np

def concurrence_via_eps(psi):
    # psi: shape (2,2) complex, psi[a,b] are amplitudes
    eps = np.array([[0,1],[-1,0]], dtype=complex)
    psi_conj = np.conjugate(psi)
    # contract: psi_{a,b} eps_{a,a'} eps_{b,b'} psi^*_{a',b'}
    val = 0+0j
    for a in range(2):
        for b in range(2):
            for ap in range(2):
                for bp in range(2):
                    val += psi[a,b] * eps[a,ap] * eps[b,bp] * psi_conj[ap,bp]
    return abs(val)

def concurrence_via_sigma_y(psi):
    # flatten psi to vector in computational basis ordering (00,01,10,11)
    vec = psi.reshape(4)
    sy = np.array([[0,-1j],[1j,0]], dtype=complex)
    K = np.kron(sy, sy)
    # psi* is complex conjugate of the vector (not transpose)
    val = np.vdot(vec, K.dot(np.conjugate(vec)))
    return abs(val)

# test on random normalized two-qubit state
v = np.random.randn(4) + 1j * np.random.randn(4)
v = v / np.linalg.norm(v)
psi = v.reshape(2,2)

c1 = concurrence_via_eps(psi)
c2 = concurrence_via_sigma_y(psi)
print(c1, c2)