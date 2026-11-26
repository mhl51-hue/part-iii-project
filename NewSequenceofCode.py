'''
Applying the wall-Chebyshev projector factor by factor with a (sampled) Hamiltonian
'''

import numpy as np

from FermiHubbard import half_filling_indices
from ApplyingProjector import estimate_energy_and_range
from SuccessProbability import lcu_alpha, wallcheb_nodes_scaled

def evaulate(sampled_H, subspace, use_exact_S, m, factors_per_sample=1, mapper=None):
    '''
    returns p, energy and fidelity of the output state, measured against the true Hamiltonian and groundstate,
    and the alpha_v used in each factor

    sampled_H: function returning a new sampled H each call
    factors_per_sample: how many consecutive factors share one sampled hamiltonian
    mapper: None if sampled_H already is a qubit Hamiltonian (Pauli sampling)
            JW mapping if sampled_H is a Fermionic0p
    '''
    # full problem in the 16 dimensional space
    H_full = subspace.qubit_H.to_matrix(sparse=True).toarray()
    keep = half_filling_indices(H_full)
    psi = np.zeros(16); psi[keep[subspace.neel]] = 1.0 # Neel determinant
    true_ground = np.zeros(16); true_ground[keep] = subspace.V[:,0] # true ground state
    S, R = estimate_energy_and_range(subspace, use_exact_S)
    I = np.eye(16)

    # the m nodes a_v of the wall-Chebyshev polynomial in the H_tilde variable
    nodes = wallcheb_nodes_scaled(m)

    # apply the factors one at a time
    alphas = np.zeros(m) # alpha_v used in each factor
    for v in range(m):
        if v % factors_per_sample == 0: # time for a new sampled H
            H_s = sampled_H()
            if mapper is not None: # fermionic sampling: second quantised H is mapped now
                H_s = mapper.map(H_s)
            H_s_qubit = H_s
            H_s = H_s.to_matrix(sparse=True).toarray()
            H_tilde = 2 * (H_s - S * I) / R - I

        # LCU 1-norm of Hs_tilde - av from this factor's sampled H
        alphas[v] = lcu_alpha(H_s_qubit, 2/R, -(2*S/R + 1 + nodes[v]))
        psi = (H_tilde @ psi - nodes[v] * psi) / alphas[v]

    p = np.vdot(psi, psi).real

    return dict(p=p, energy=np.vdot(psi, H_full @ psi).real / p, fidelity=abs(np.vdot(true_ground, psi)) / np.sqrt(p),
                alphas=alphas)

def average_over_runs_new(sampled_H, subspace, use_exact_S, m, factors_per_sample, n_runs=100, mapper=None):
    '''
    p and fideility are computed for every run and then averaged
    p: geometric mean over runs
    p_mean: arithmetic mean (expected success probability of the protocol)
    '''
    runs = [evaulate(sampled_H, subspace, use_exact_S, m, factors_per_sample, mapper) for _ in range(n_runs)]
    p = np.array([r["p"] for r in runs])
    F = np.array([r["fidelity"] for r in runs])
    return dict(p=float(np.exp(np.mean(np.log(p)))), p_mean=p.mean(), F_mean=float(np.nanmean(F)), F_std=float(np.nanstd(F)), F_runs=F, p_runs=p)
