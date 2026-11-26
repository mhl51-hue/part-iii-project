"""
Compares Hamiltonian sampling methods for the wall-Chebyshev ground state preparation
of the two-site Fermi-Hubbard model

Produces two figures, each showing fidelity with the true ground state and success
probability against polynomial order m
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit_nature.second_q.mappers import JordanWignerMapper

from FermiHubbard import fermihubbard_subspace, second_quantised_H
from DeterministicSampling import pauli_truncate, evaluate
from PauliSampling import (rand_pauli_sampler_with_replacement, rand_pauli_sampler_without_replacement,
                            hybrid_sampling)
from NewSequenceofCode import average_over_runs_new
from FermionicSampling import (fermionic_groups, rand_fermionic_sampler_with_replacement,
                               rand_fermionic_sampler_without_replacement, fermionic_hybrid_sampling)


'''
Setting different parameters
'''
U = 2 # on-site interaction strength
exact_S = True # True: S = E0, False: S = Hartree-Fock energy
m_vals = np.arange(1, 25) # orders of the polynomial expansion
n_runs = 50 # number of independent runs for each stochastic method
per_sample = 1 # number of factors applied per sampled Hamiltonian

# parameters for pauli
n_keep_p = 7 # terms kept in deterministic truncation and random sampling without replacement
n_draws_p = 11 # terms drawn in random sampling with replacement
n_keep_hybrid_p = 4 # hybrid sampling: heaviest terms always kept
n_draws_hybrid_p = 4 # plus this many drawn at random from the rest

'''
Fermionic sampling
hamiltonian has 4 fermionic terms
hopping up
hopping down
on site 0
on site 1
'''
n_terms_f = 2 # terms kept in random sampling without replacement
n_draws_f = 3 # terms drawn in random sampling with replacement
n_keep_hybrid_f = 1 # hybrid sampling: heaviest terms always kept
n_draws_hybrid_f = 1 # plus this many drawn at random from the rest

def scan(H):
    '''
    evaluates a single fixed Hamiltonian at every m
    returns (fidelity, p_succ), each an array of shape (len(m_vals),)
    '''
    # deg only affects the step function
    results = [evaluate(H, subspace, exact_S, m=int(m), deg=50)["wall_cheb_product"] for m in m_vals]
    return np.array([r["fidelity"] for r in results]), np.array([r["p"] for r in results])

def scan_runs_new(sampled_H, n_runs, mapper=None):
    '''
    for each m, n_runs independent runs, each applying m factors with a
    newly sampled Hamiltonian every per_sample factors
    '''
    F, P = [], []
    for m in m_vals:
        out = average_over_runs_new(sampled_H, subspace, exact_S, int(m),
                                    factors_per_sample=per_sample, n_runs=n_runs, mapper=mapper)
        F.append(out["F_runs"])
        P.append(out["p_runs"])
    return np.array(F).T, np.array(P).T


def summarise_geometric(F, P):
    '''
    returns (mean fidelity, std of fidelity), (geometric mean of p_succ, std of log p_succ)
    '''
    log_P = np.log(np.where(P > 0, P, np.nan))
    return (np.nanmean(F, 0), np.nanstd(F, 0)), (np.exp(np.nanmean(log_P, 0)), np.nanstd(log_P, 0))


def summarise_arithmetic(F, P):
    '''
    returns (mean fidelity, std of fidelity), (arithmetic mean of p_succ, standard error of that mean)
    '''
    n = np.sum(np.isfinite(P), 0)
    return (np.nanmean(F, 0), np.nanstd(F, 0)), (np.nanmean(P, 0), np.nanstd(P, 0, ddof=1) / np.sqrt(n))


def strings_per_sample(sampled_H, mapper=None, n=200):
    '''
    mean number of distinct non-identity Pauli strings in a sampled H
    '''
    def n_strings(H):
        if mapper is not None:
            H = mapper.map(H)
        return sum(1 for label, c in H.simplify().to_list() if set(label) != {"I"} and abs(c) > 1e-12)
    return float(np.mean([n_strings(sampled_H()) for _ in range(n)]))


def plot_comparison(series, title, filename, legend_fontsize):
    '''
    plots fidelity and p_succ against m
    '''
    fig, (ax_F, ax_p) = plt.subplots(1, 2, figsize=(10, 4))

    for label, F_curve, p_curve, colour, linestyle in series:
        ax_F.plot(m_vals, F_curve, linestyle, color=colour, label=label)
        ax_p.plot(m_vals, p_curve, linestyle, color=colour, label=label)

    # Fidelity plot
    ax_F.legend(fontsize=legend_fontsize, loc="lower left")
    ax_F.axhline(1.0, color="k", ls="--", lw=0.8)
    ax_F.set_ylim(0, 1.05)
    ax_F.set_ylabel("fidelity with true ground state")

    # Success probability plot
    ax_p.set_yscale("log")
    ax_p.set_ylim(*p_lim)
    ax_p.set_ylabel(r"$p_{\rm succ}$")

    for ax in (ax_F, ax_p):
        ax.set_xlabel("polynomial order m")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)


'''
Hamiltonians
'''

subspace = fermihubbard_subspace(U)
H_full = subspace.qubit_H
H_truncated = pauli_truncate(H_full, n_keep_p)

n_strings = len(H_full.simplify().to_list()) - 1 # number of non-identity Pauli strings in H_full


'''
Pauli sampling
'''

# Full Hamiltonian and deterministic truncation
F_full, p_full = scan(H_full)
F_truncated, p_truncated = scan(H_truncated)

# Random sampling with replacement
with_rep_H = rand_pauli_sampler_with_replacement(H_full, n_draws=n_draws_p, seed=0)
(F_mean_with_rep, F_std_with_rep), (p_with_rep, p_sem_with_rep) = summarise_arithmetic(*scan_runs_new(with_rep_H, n_runs))

# Random sampling without replacement, reweighted
no_rep_H = rand_pauli_sampler_without_replacement(H_full, n_terms=n_keep_p, reweight=True, seed=0)
(F_mean_no_rep, F_std_no_rep), (p_no_rep, p_sem_no_rep) = summarise_arithmetic(*scan_runs_new(no_rep_H, n_runs))

# Hybrid sampling
hybrid_H = hybrid_sampling(H_full, n_keep=n_keep_hybrid_p, n_draws=n_draws_hybrid_p, seed=0)
(F_mean_hybrid, F_std_hybrid), (p_hybrid, p_sem_hybrid) = summarise_arithmetic(*scan_runs_new(hybrid_H, n_runs))


'''
Fermionic Sampling
'''

groups = fermionic_groups(second_quantised_H(U), JordanWignerMapper())

# Random sampling with replacement
with_rep_f = rand_fermionic_sampler_with_replacement(groups, n_draws=n_draws_f, seed=0)
(F_mean_with_rep_f, _), (p_with_rep_f, _) = summarise_arithmetic(*scan_runs_new(with_rep_f, n_runs, mapper=groups.mapper))

# Random sampling without replacement, reweighted
no_rep_f = rand_fermionic_sampler_without_replacement(groups, n_terms=n_terms_f, reweight=True, seed=0)
(F_mean_no_rep_f, _), (p_no_rep_f, _) = summarise_arithmetic(*scan_runs_new(no_rep_f, n_runs, mapper=groups.mapper))

# Hybrid sampling
hybrid_f = fermionic_hybrid_sampling(groups, n_keep=n_keep_hybrid_f, n_draws=n_draws_hybrid_f, seed=0)
(F_mean_hybrid_f, _), (p_hybrid_f, _) = summarise_arithmetic(*scan_runs_new(hybrid_f, n_runs, mapper=groups.mapper))

full_H_series = ("full Hamiltonian", F_full, p_full, "gray", "--")

series_pauli = [
    full_H_series,
    #(f"deterministic truncation, n = {n_keep_p}",
     #F_truncated, p_truncated, "C0", "-"),
    (f"random, with replacement ({n_draws_p} draws, {strings_per_sample(with_rep_H):.1f} strings)",
     F_mean_with_rep, p_with_rep, "C2", "-"),
    (f"random, no replacement, reweighted (n = {n_keep_p}, {strings_per_sample(no_rep_H):.1f} strings)",
     F_mean_no_rep, p_no_rep, "red", "-"),
    (f"hybrid sampling (n = {n_keep_hybrid_p}, + {n_draws_hybrid_p} draws, {strings_per_sample(hybrid_H):.1f} strings)",
     F_mean_hybrid, p_hybrid, "orange", "-"),
]

series_fermionic = [
    full_H_series,
    (f"random, with replacement ({n_draws_f} draws, {strings_per_sample(with_rep_f, groups.mapper):.1f} strings)",
     F_mean_with_rep_f, p_with_rep_f, "C2", "-"),
    (f"random, no replacement, reweighted ({n_terms_f} terms, {strings_per_sample(no_rep_f, groups.mapper):.1f} strings)",
     F_mean_no_rep_f, p_no_rep_f, "red", "-"),
    (f"hybrid sampling ({n_keep_hybrid_f} + {n_draws_hybrid_f} draws, {strings_per_sample(hybrid_f, groups.mapper):.1f} strings)",
     F_mean_hybrid_f, p_hybrid_f, "orange", "-"),
]

# p_succ axis
p_all = np.concatenate([p_full, p_truncated])
p_lim = (10 ** np.floor(np.log10(p_all.min())), 1)


'''
Plotting the figures
'''

if __name__ == "__main__":
    S_label = "E0" if exact_S else "HF"

    plot_comparison(series_pauli,
                    title=f"wall-Chebyshev, U = {U}, S = {S_label}, Pauli sampling, {n_runs} runs for stochastic methods",
                    filename="ComparingSampling_pauli.png",
                    legend_fontsize=8)

    plot_comparison(series_fermionic,
                    title=f"wall-Chebyshev, U = {U}, S = {S_label}, fermionic sampling, "
                          f"{n_runs} runs for stochastic methods",
                    filename="ComparingSampling_fermionic.png",
                    legend_fontsize=7)
