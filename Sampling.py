'''
Sampling the terms of the Hamiltonian
1. defining the truncation and % of hamiltonian functions
2. pauli sampling
3. second quantised sampling
4. evaluating the projectors with the sampled hamiltonian
'''

import numpy as np
from numpy.polynomial import chebyshev as Cheb
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.quantum_info import SparsePauliOp

from FermiHubbard import half_filling_indices
from ApplyingProjector import estimate_energy_and_range
from Projectors import (
    wall_cheb_coefs,
    delta_func,
    step_func,
    step
    )
from SuccessProbability import lcu_alpha
from FermiHubbard import fermihubbard_subspace

'''
1. Giving a weight to each term to then be able to decide which terms to keep
'''

# Truncation method
def keep_n_largest(weights, n):
    '''
    Indices of the n terms with the largest weight
    Allows for truncating the sum at a certain n
    '''
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(-weights, kind="stable")[:n]
    return order

# By percentage of the full Hamiltonian
def keep_weight_fraction(weights, fraction):
    '''
    Indices of the weights of the terms that add up the a certain
    percentage of the total Hamiltonian
    Allows for truncation at a set weight fraction of terms
    '''
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(-weights, kind="stable")
    cumulative = np.cumsum(weights[order])
    return order[: np.searchsorted(cumulative, fraction*cumulative[-1] - 1e-12)+1]

'''
2. Pauli hamiltonian sampling
Post JW mapping the terms are pauli strings with real coefficients
Each string is hermitian
The identity string is never sampled as it is a constant energy shift 
'''

def pauli_terms(op):
    '''
    spliting SparsePauliOp into identity terms and non identity terms
    '''
    ident, terms = [], []
    for label, coef in op.simplify().to_list():
        (ident if set(label) == {"I"} else terms).append((label, coef))
    return ident, terms

def pauli_rebuild(ident, terms, keep):
    return SparsePauliOp.from_list(ident + [terms[i] for i in keep])

def pauli_truncate(op, n):
    '''
    Keeping the first n terms in the pauli string with the largest abs coefficients
    '''
    ident, terms = pauli_terms(op)
    pauli_truncated = pauli_rebuild(ident, terms, keep_n_largest([abs(c) for _, c in terms], n))
    return pauli_truncated

def pauli_weight(op, fraction):
    '''
    Keeping a fraction of the full hamiltonian
    '''
    ident, terms = pauli_terms(op)
    pauli_weight = pauli_rebuild(ident, terms, keep_weight_fraction([abs(c) for _, c in terms], fraction))
    return pauli_weight

'''
3. Second quantised hamiltonian sampling
'''

'''
4. Evaluation
'''

def evaluate(sampled_op, subspace, ground_state_energy, m=10, deg=1000, epsilon=1e-2):
    '''
    - p, energy and ground state fidelity of projectors comes from the new sampled hamiltonian
    - S and R come from full hamiltonian
    - alpha is the LCU 1-norm of the sampled operator
    - energy and fidelity are measured against the full hamiltonian and ground state
    - now using the full 16x16 space
    '''
    # Full problem embedded in the 16x16 space
    H_full = subspace.qubit_H.to_matrix(sparse=True).toarray()
    keep = half_filling_indices(H_full)
    psi = np.zeros(16); psi[keep[subspace.neel]] = 1.0 # Neel determinant
    ground = np.zeros(16); ground[keep] = subspace.V[:, 0]
    S, R = estimate_energy_and_range(subspace, ground_state_energy)

    # the sampled hamiltonian, its eigenstates and the neel state's overlap with them
    E, V = np.linalg.eigh(sampled_op.to_matrix(sparse=True).toarray())
    c = V.conj().T @ psi

    # alpha of each projector using sampled H
    alpha_wall = lcu_alpha(sampled_op, 2/R, -(2*S/R+1))
    alpha_stepdelta = lcu_alpha(sampled_op, shift=-S)
    g = subspace.gap / alpha_stepdelta
    coefs, _ = step_func(deg, delta=g / 4, mu=g / 2, epsilon=epsilon)
    coefs /= max(1.0, np.max(np.abs(step(np.linspace(-1, 1, 4000), coefs))))

    weights = {
        "wall_cheb": Cheb.chebval((2 * (E - S) / R - 1) / alpha_wall, wall_cheb_coefs(m)),
        "delta_func": delta_func((E - S) / alpha_stepdelta, m // 2, g),
        "step_func": step((E - S) / alpha_stepdelta, coefs),
    }

    out = dict(alpha_wall=alpha_wall, alpha_stepdelta=alpha_stepdelta, n_pauli=len(sampled_op.simplify().to_list()))
    for name, w in weights.items():
        psi_out = V @ (w * c)
        p = np.vdot(psi_out, psi_out).real
        out[name] = (dict(p=p, energy=np.vdot(psi_out, H_full @ psi_out).real / p,
                        fidelity=abs(np.vdot(ground, psi_out)) / np.sqrt(p))
                    if p > 1e-300 else dict(p=p, energy=np.nan, fidelity=np.nan))
    return out

if __name__ == "__main__":
    mapper = JordanWignerMapper()
    U = 4
    subspace = fermihubbard_subspace(U)
    n_pauli = len(subspace.qubit_H.simplify().to_list()) - 1 # excludes identity term
    print(f"\nU={U}: {n_pauli} non-identity pauli terms")

    samples = [("no sampling", subspace.qubit_H)]
    samples += [(f"truncate n={n}", pauli_truncate(subspace.qubit_H, n))for n in range(n_pauli - 1, 0, -2)]
    samples += [(f"fraction f={f:.0%}", pauli_weight(subspace.qubit_H, f)) for f in (0.9, 0.7, 0.5)]
    exact = True
    for name, op in samples:
        r = evaluate(op, subspace, exact)
        print(f"{name:15s} S={'E0' if exact else 'HF'}  n_Pauli={r['n_pauli']:2d} | "
                        + "  ".join(f"{k}: success_p={r[k]['p']:.2e} Fidelity={r[k]['fidelity']:.3f}" for k in ("wall_cheb", "delta_func", "step_func")))
                    

