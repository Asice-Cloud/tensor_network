#!/usr/bin/env python3
"""
Ising / Majorana anyon demo

 - Construct 4 Majorana operators as 4x4 matrices (Pauli tensor representation)
 - Build braid unitaries U_{i,i+1} = exp( (pi/4) γ_i γ_{i+1} ) = (1/√2)(I + γ_i γ_{i+1})
 - Verify these unitaries are in the Clifford group by checking conjugation maps Pauli strings to Pauli strings

Usage: python3 ising_anyons.py
"""

#通过泡利矩阵张量积表示构造Majorana算符
#构造辫子单元 U_{i,i+1} = exp( (pi/4) γ_i γ_{i+1} ) = (1/√2)(I + γ_i γ_{i+1})
#验证这些单元在Clifford群中,通过检查共轭作用将泡利字符串映射到泡利字符串
import numpy as np


def kron(*mats):
    out = mats[0]
    for M in mats[1:]:
        out = np.kron(out, M)
    return out


def paulis():
    I = np.array([[1,0],[0,1]], dtype=complex)
    X = np.array([[0,1],[1,0]], dtype=complex)
    Y = np.array([[0,-1j],[1j,0]], dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)
    return I, X, Y, Z


def majorana_mapping():
    # mapping used in earlier discussion:
    # γ1 = σ_x ⊗ I
    # γ2 = σ_y ⊗ I
    # γ3 = σ_z ⊗ σ_x
    # γ4 = σ_z ⊗ σ_y
    I, X, Y, Z = paulis()
    g1 = kron(X, I)
    g2 = kron(Y, I)
    g3 = kron(Z, X)
    g4 = kron(Z, Y)
    return [g1, g2, g3, g4]


def braid_unitary(g_i, g_j):
    # Using (γ_i γ_j)^2 = -I, so exp(theta * γiγj) = cos(theta) I + sin(theta) γiγj
    theta = np.pi / 4
    gij = g_i @ g_j
    return np.cos(theta) * np.eye(gij.shape[0], dtype=complex) + np.sin(theta) * gij


def all_pauli_strings():
    I, X, Y, Z = paulis()
    labels = ['I','X','Y','Z']
    mats = []
    labs = []
    for a,label_a in zip([I,X,Y,Z], labels):
        for b,label_b in zip([I,X,Y,Z], labels):
            mats.append(kron(a,b))
            labs.append(label_a + label_b)
    return mats, labs


def find_pauli_equiv(M, paulis, labels, tol=1e-8):
    # Find if M is proportional to one of paulis[k] up to a scalar
    for k, P in enumerate(paulis):
        # compute scalar s such that M ≈ s P
        s = np.vdot(P.flatten(), M.flatten()) / np.vdot(P.flatten(), P.flatten())
        if np.linalg.norm(M - s * P) < tol:
            return True, k, s
    return False, None, None


def main():
    gammas = majorana_mapping()
    U12 = braid_unitary(gammas[0], gammas[1])
    U23 = braid_unitary(gammas[1], gammas[2])
    U34 = braid_unitary(gammas[2], gammas[3])

    print('Constructed Majorana γ matrices (4x4).')
    print('U12, U23, U34 are 4x4 unitaries implementing braids.')

    pauli_mats, pauli_labels = all_pauli_strings()

    # Check unitarity
    print('\nUnitarity checks:')
    print('||U12†U12 - I|| =', np.linalg.norm(U12.conj().T @ U12 - np.eye(4)))
    print('||U23†U23 - I|| =', np.linalg.norm(U23.conj().T @ U23 - np.eye(4)))
    print('||U34†U34 - I|| =', np.linalg.norm(U34.conj().T @ U34 - np.eye(4)))

    # For each braid unitary, check conjugation maps Pauli strings to Pauli strings
    for name, U in [('U12', U12), ('U23', U23), ('U34', U34)]:
        print(f'\nChecking conjugation by {name}:')
        all_ok = True
        for P, lab in zip(pauli_mats, pauli_labels):
            M = U @ P @ U.conj().T
            ok, idx, s = find_pauli_equiv(M, pauli_mats, pauli_labels)
            if not ok:
                print('  NOT a Pauli after conjugation:', lab)
                all_ok = False
                break
        if all_ok:
            print('  All Pauli strings map to Pauli strings (up to scalar) — this is a Clifford operation.')

    # Show explicit action on single-qubit logical operators (example)
    print('\nExample conjugations on single-qubit Paulis (tensor form):')
    I, X, Y, Z = paulis()
    X1 = kron(X, I)
    Z1 = kron(Z, I)
    X2 = kron(I, X)
    Z2 = kron(I, Z)
    print('U12 X1 U12† =')
    print(np.round(U12 @ X1 @ U12.conj().T, 6))
    print('U12 Z1 U12† =')
    print(np.round(U12 @ Z1 @ U12.conj().T, 6))

    # Eigenvalues (showing discrete phases)
    ev12, _ = np.linalg.eig(U12)
    ev23, _ = np.linalg.eig(U23)
    print('\nEigenvalues U12:', np.round(ev12, 6))
    print('Eigenvalues U23:', np.round(ev23, 6))

    print('\nConclusion: braid unitaries from Majorana (Ising anyons) are Clifford (they normalize the Pauli group).')
    print('Therefore, braiding Majorana modes alone cannot generate a universal (non-Clifford) gate set; extra non-topological resources are required (e.g., magic-state injection).')


if __name__ == '__main__':
    main()
