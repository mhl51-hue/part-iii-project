import numpy as np
import matplotlib.pyplot as plt
 
from FermiHubbard import fermihubbard_subspace
from DeterministicSampling import pauli_terms, evaluate
from PauliSampling import rand_pauli_sampler_without_replacement
from NewSequenceofCode import average_over_runs_new
 
U = 2
n_terms = 6
n_runs = 50
per_sample = 1                       
show_bands = True
band_width = (16, 84)                                 
exact_s = True
m_vals = np.arange(1, 25)
 
subspace = fermihubbard_subspace(U)
op = subspace.qubit_H
sample_H = rand_pauli_sampler_without_replacement(op, n_terms, reweight=True, seed=0)
 
 
def scan(h):
    '''fidelity and p of the full Hamiltonian at every order m'''
    res = [evaluate(h, subspace, exact_s, m=int(m), deg=50)["wall_cheb_product"] for m in m_vals]
    return np.array([r["fidelity"] for r in res]), np.array([r["p"] for r in res])
 
 
def scan_chain(sampler):
    '''
    for each order m: n_runs independent runs, each applying m factors with a fresh sampled H every per_sample factors
    returns arrays of shape (n_runs, len(m_vals))
    '''
    F, P = [], []
    for m in m_vals:
        m = int(m)
        out = average_over_runs_new(sampler, subspace, exact_s, m, factors_per_sample=min(per_sample, m), n_runs=n_runs)
        F.append(out["F_runs"]); P.append(out["p_runs"])
    return np.array(F).T, np.array(P).T
 
 
F_full, p_full = scan(op)
F, P = scan_chain(sample_H)
n_distinct = np.mean([len(pauli_terms(sample_H())[1]) for _ in range(200)])
logp = np.log10(P)
 
fig, (ax_F, ax_p) = plt.subplots(1, 2, figsize=(11, 4.5))
 
ax_F.plot(m_vals, F_full, "--", color="gray", label="full Hamiltonian (10 strings)")
ax_F.plot(m_vals, F.mean(0), color="red", label=f"random, no replacement, reweighted ({n_distinct:.1f} strings on average)")
ax_F.axhline(1, color="k", ls="--", lw=0.6)
ax_F.set_ylim(0, 1.05)
ax_F.set_xlabel("polynomial order m")
ax_F.set_ylabel("fidelity with true ground state")
ax_F.legend(fontsize=7, loc="lower right")
 
ax_p.plot(m_vals, p_full, "--", color="gray")
ax_p.plot(m_vals, 10 ** logp.mean(0), color="red")
ax_p.set_yscale("log")
ax_p.set_xlabel("polynomial order m")
ax_p.set_ylabel(r"$p_{\rm succ}$")
 
if show_bands:
    ax_F.fill_between(m_vals, *np.percentile(F, band_width, axis=0), color="red", alpha=0.15)
    ax_p.fill_between(m_vals, *10 ** np.percentile(logp, band_width, axis=0), color="red", alpha=0.15)
 
fig.suptitle(f"wall-Chebyshev (product of LCUs), U = {U}: random sampling without replacement, {n_terms} terms, {n_runs} runs, new H every {per_sample} factor(s)", fontsize=10)
fig.tight_layout()
fig.savefig("errorandsshown.png", dpi=200)
plt.show()
 
