import numpy as np
import matplotlib.pyplot as plt
 
from FermiHubbard import fermihubbard_subspace
from DeterministicSampling import pauli_terms, evaluate
from Randomsampling import rand_pauli_sampler_without_replacement
from NewSequenceofCode import average_over_runs_new
 
U = 2
N_TERMS = 6
N_RUNS = 50
PER_SAMPLE = 1                       
SHOW_BANDS = True
BAND = (16, 84)                                 
EXACT_S = True
M_VALUES = np.arange(1, 25)
 
subspace = fermihubbard_subspace(U)
op = subspace.qubit_H
sample_H = rand_pauli_sampler_without_replacement(op, N_TERMS, reweight=True, seed=0)
 
 
def scan(h):
    '''fidelity and p of the full Hamiltonian at every order m'''
    res = [evaluate(h, subspace, EXACT_S, m=int(m), deg=50)["wall_cheb"] for m in M_VALUES]
    return np.array([r["fidelity"] for r in res]), np.array([r["p"] for r in res])
 
 
def scan_chain(sampler):
    '''
    for each order m: N_RUNS independent runs, each applying m factors with a fresh sampled H every PER_SAMPLE factors
    returns arrays of shape (N_RUNS, len(M_VALUES))
    '''
    F, P = [], []
    for m in M_VALUES:
        m = int(m)
        out = average_over_runs_new(sampler, subspace, EXACT_S, m, factors_per_sample=min(PER_SAMPLE, m), n_runs=N_RUNS)
        F.append(out["F_runs"]); P.append(out["p_runs"])
    return np.array(F).T, np.array(P).T
 
 
F_full, p_full = scan(op)
F, P = scan_chain(sample_H)
n_distinct = np.mean([len(pauli_terms(sample_H())[1]) for _ in range(200)])
logp = np.log10(P)
 
fig, (ax_F, ax_p) = plt.subplots(1, 2, figsize=(11, 4.5))
 
ax_F.plot(M_VALUES, F_full, "--", color="gray", label="full Hamiltonian (10 strings)")
ax_F.plot(M_VALUES, F.mean(0), color="red", label=f"random, no replacement, reweighted ({n_distinct:.1f} strings on average)")
ax_F.axhline(1, color="k", ls="--", lw=0.6)
ax_F.set_ylim(0, 1.05)
ax_F.set_xlabel("polynomial order m")
ax_F.set_ylabel("fidelity with true ground state")
ax_F.legend(fontsize=7, loc="lower right")
 
ax_p.plot(M_VALUES, p_full, "--", color="gray")
ax_p.plot(M_VALUES, 10 ** logp.mean(0), color="red")
ax_p.set_yscale("log")
ax_p.set_xlabel("polynomial order m")
ax_p.set_ylabel(r"$p_{\rm succ}$")
 
if SHOW_BANDS:
    ax_F.fill_between(M_VALUES, *np.percentile(F, BAND, axis=0), color="red", alpha=0.15)
    ax_p.fill_between(M_VALUES, *10 ** np.percentile(logp, BAND, axis=0), color="red", alpha=0.15)
 
fig.suptitle(f"wall-Chebyshev, U = {U}: random sampling without replacement, {N_TERMS} terms, {N_RUNS} runs, new H every {PER_SAMPLE} factor(s)", fontsize=10)
fig.tight_layout()
fig.savefig("errorandsshown.png", dpi=200)
plt.show()
 
