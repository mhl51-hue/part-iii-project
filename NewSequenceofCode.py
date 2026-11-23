import numpy as np

from FermiHubbard import half_filling_indices
from ApplyingProjector import estimate_energy_and_range
from SuccessProbability import lcu_alpha

def evaulate(sampled_H, subspace, use_exact_S, m, factors_per_sample=1, alpha=None):
    '''
    returns p, energy and fidelity of the output state, measured against the true Hamiltonian and groundstate

    sampled_H: function returning a new sampled H each call
    factors_per_sample: how many consecutive factors share one sampled hamiltonian
    alpha: alpha used in every factor (ie smaller for deterministic truncation), if = None then it is the alpha
    of the true hamiltonian which is correct for the unbiased samplers
    '''
    # full problem in the 16 dimensional space
    H_full = subspace.qubit_H.to_matrix(sparse=True).toarray()
    keep = half_filling_indices(H_full)
    psi = np.zeros(16); psi[keep[subspace.neel]] = 1.0 # Neel determinant
    true_ground = np.zeros(16); true_ground[keep] = subspace.V[:,0] # true ground state
    S, R = estimate_energy_and_range(subspace, use_exact_S)
    if alpha is None:
        alpha = lcu_alpha(subspace.qubit_H, 2/R, -(2*S/R+1))
    I = np.eye(16)

    # the m modes, and the normilisation for wall_cheb
    nodes = -np.cos(np.arange(1, m+1) * np.pi / (m+0.5))
    y = np.linspace(-1, 1, 2001)
    norm = np.max(np.abs(np.prod(y[:, None] - nodes / alpha, axis=1)))

    # apply the factors one at a time
    for v in range(m):
        if v % factors_per_sample == 0: # time for a new sampled H
            H_s = sampled_H().to_matrix(sparse=True).toarray()
            H_tilde = 2 * (H_s - S * I) / R - I
        psi = (H_tilde @ psi - nodes[v] * psi) / alpha
    psi = psi / norm

    p = np.vdot(psi, psi).real

    return dict(p=p, energy=np.vdot(psi, H_full @ psi).real / p, fidelity=abs(np.vdot(true_ground, psi)) / np.sqrt(p), alpha=alpha)

def average_over_runs_new(sampled_H, subspace, use_exact_S, m, factors_per_sample, n_runs=100, alpha=None):
    '''
    p and fideility are computed for every run and then averaged
    p is a geometric mean
    '''
    runs = [evaulate(sampled_H, subspace, use_exact_S, m, factors_per_sample, alpha) for _ in range(n_runs)]
    p = np.array([r["p"] for r in runs])
    F = np.array([r["fidelity"] for r in runs])
    return dict(p=float(np.exp(np.mean(np.log(p)))), p_mean=p.mean(), F_mean=float(np.nanmean(F)), F_std=float(np.nanstd(F)), F_runs=F, p_runs=p)
