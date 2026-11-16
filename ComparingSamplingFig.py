import numpy as np
import matplotlib.pyplot as plt
 
from FermiHubbard import fermihubbard_subspace
from DeterministicSampling import pauli_truncate, evaluate
from Randomsampling import rand_pauli_sampler_with_replacement, rand_pauli_sampler_without_replacement, hybrid_sampling

'''
Variables in each sampling method, want all methods to have 5 distinct terms in hamiltonian
n_keep = 5, used in deterministic truncated H as well as random sampling without replacing terms
n_draws = 5, used in random with replacement

n_keep_hybrid = 2
n_draws_hybrid = 3 hybrid sampling uses a mix of kept terms and drawn terms

n_runs = 100, the amount of times the stochastic samples are run
'''

U = 2
exact_S = True # True: S = E0, False: S = Hartree-Fock energy
m_vals = np.arange(1, 25) # Order of polynomial expansion

n_keep = 5 
n_draws = 7
n_keep_hybrid = 3
n_draws_hybrid = 2

n_runs = 100
 
subspace = fermihubbard_subspace(U)
truncated = pauli_truncate(subspace.qubit_H, n_keep)
H_full = subspace.qubit_H
n_strings = len(H_full.simplify().to_list()) - 1
 
def scan(H):
    results = [evaluate(H, subspace, exact_S, m=int(m), deg=50)["wall_cheb"] for m in m_vals]    # deg only affects the step function
    return np.array([r["fidelity"] for r in results]), np.array([r["p"] for r in results])
 
def scan_runs(sample_H, n_runs):
    '''
    one random hamiltonian per run each scanned over all m
    '''
    F, P = zip(*[scan(sample_H()) for _ in range(n_runs)])
    return np.array(F), np.array(P)

def summarise(F, P):
    '''
    mean fidelity and its spread
    geometric mean P and its multiplication spread
    '''
    logp = np.log(np.where(P > 0, P, np.nan))
    return (np.nanmean(F, 0), np.nanstd(F, 0)), (np.exp(np.nanmean(logp, 0)), np.nanstd(logp, 0))

# Full hamiltonian and truncated hamiltonian
curves = {"full hamiltonian": scan(subspace.qubit_H), "truncated hamiltonian": scan(truncated)}

# Stochastic sampling with replacement of terms
with_replacement_H = rand_pauli_sampler_with_replacement(H_full, n_draws=n_draws, seed=0)
(F_mean_with_rep, F_std_with_rep), (p_with_rep, p_logstd_with_rep) = summarise(*scan_runs(with_replacement_H, n_runs))

# Stochastic sampling without replacement of terms
without_replacement_H = rand_pauli_sampler_without_replacement(H_full, n_terms=n_keep, reweight=True, seed=0)
(F_mean_no_rep, F_std_no_rep), (p_no_rep, p_logstd_no_rep) = summarise(*scan_runs(without_replacement_H, n_runs))

# Hybrid sampling
hybrid_H = hybrid_sampling(H_full, n_keep=n_keep_hybrid, n_draws=n_draws_hybrid, seed=0)
(F_hybrid, F_std_hybrid), (p_hybrid, p_logstd_hybrid) = summarise(*scan_runs(hybrid_H, n_runs))

p_all = np.concatenate([p for _, p in curves.values()])
p_lim = (10 ** np.floor(np.log10(p_all.min())), 1)                      # same p axis on both figures

if __name__ == "__main__":
    fig, (ax_F, ax_p) = plt.subplots(1, 2, figsize=(10, 4))

    # Full hamiltonian
    ax_F.plot(m_vals, curves["full hamiltonian"][0], "--", color="gray", label="full Hamiltonian")
    ax_p.plot(m_vals, curves["full hamiltonian"][1], "--", color="gray", label="full Hamiltonian")
    # Deterministic truncation
    ax_F.plot(m_vals, curves["truncated hamiltonian"][0], "-", color="C0", label=f"deterministic truncation, n = {n_keep}")
    ax_p.plot(m_vals, curves["truncated hamiltonian"][1], "-", color="C0", label=f"deterministic truncation, n = {n_keep}")
    # Rand sampling with replacement
    ax_F.plot(m_vals, F_mean_with_rep, "-", color="C2", label=f"random sampling with replacement ({n_draws} draws)")
    ax_p.plot(m_vals, p_with_rep, "-", color="C2", label=f"random sampling with replacement ({n_draws} draws)")
    # Rand sampling without replacement
    ax_F.plot(m_vals, F_mean_no_rep, "-", color="pink", label=f"random sampling without replacement (n = {n_keep})")
    ax_p.plot(m_vals, p_no_rep, "-", color="pink", label=f"random sampling without replacement (n = {n_keep})")
    # Hybrid sampling
    ax_F.plot(m_vals, F_hybrid, "-", color="orange", label=f"hybrid sampling (n = {n_keep_hybrid}, + {n_draws_hybrid} draws)")
    ax_p.plot(m_vals, p_hybrid, "-", color="orange", label=f"hybrid sampling (n = {n_keep_hybrid}, + {n_draws_hybrid} draws)")

    ax_F.legend(fontsize=8, loc='lower left')
    ax_F.axhline(1.0, color="k", ls="--", lw=0.8)
    ax_F.set_ylim(0, 1.05)
    ax_F.set_ylabel(f"fidelity with true ground state, {n_runs} runs for stochastic methods")

    ax_p.set_yscale("log"); ax_p.set_ylim(*p_lim)
    ax_p.set_ylabel(r"$p_{\rm succ}$")

    for ax in (ax_F, ax_p):
        ax.set_xlabel("polynomial order m")

    fig.suptitle(f"wall-Chebyshev, U = {U}, S = {'E0' if exact_S else 'HF'}", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"ComparingSampling.png", dpi=200)
