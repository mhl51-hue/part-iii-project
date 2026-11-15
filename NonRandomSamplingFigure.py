import numpy as np
import matplotlib.pyplot as plt
 
from FermiHubbard import fermihubbard_subspace
from DeterministicSampling import pauli_truncate, evaluate
from Randomsampling import rand_pauli_sampler_with_replacement
 
U = 2
N_KEEP = 6                                      # Pauli strings kept in the truncated Hamiltonian
EXACT_S = True                                  # True: S = E0, False: S = Hartree-Fock energy
M_VALUES = np.arange(1, 50)
n_draws = 200 # will use the number of pauli strings in H
n_runs = 100
 
subspace = fermihubbard_subspace(U)
truncated = pauli_truncate(subspace.qubit_H, N_KEEP)
H = subspace.qubit_H
n_strings = len(H.simplify().to_list()) - 1
n_draws = n_strings if n_draws is None else n_draws

 
def scan(H):
    results = [evaluate(H, subspace, EXACT_S, m=int(m), deg=50)["wall_cheb"] for m in M_VALUES]    # deg only affects the step function
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


curves = {"full hamiltonian": scan(subspace.qubit_H), "truncated hamiltonian": scan(truncated)}
sample_H = rand_pauli_sampler_with_replacement(H, n_draws=n_draws, seed=0)
(F_mean, F_std), (p_geo, p_logstd) = summarise(*scan_runs(sample_H, n_runs))

p_all = np.concatenate([p for _, p in curves.values()])
p_lim = (10 ** np.floor(np.log10(p_all.min())), 1)                      # same p axis on both figures

for name, title in (("full hamiltonian", "Full Hamiltonian"), ("sampled", "Sampled Hamiltonians")):
    fig, (ax_F, ax_p) = plt.subplots(1, 2, figsize=(10, 4))
    if name == "full hamiltonian":
        ax_F.plot(M_VALUES, curves["full hamiltonian"][0], "o-", ms=3)
        ax_p.plot(M_VALUES, curves["full hamiltonian"][1], "o-", ms=3, color="C1")
    else:
        ax_F.plot(M_VALUES, curves["full hamiltonian"][0], "--", color="gray", label="full Hamiltonian")
        ax_p.plot(M_VALUES, curves["full hamiltonian"][1], "--", color="gray", label="full Hamiltonian")
        ax_F.plot(M_VALUES, curves["truncated hamiltonian"][0], "-", color="C0", label=f"deterministic truncation, n = {N_KEEP}")
        ax_p.plot(M_VALUES, curves["truncated hamiltonian"][1], "-", color="C0", label=f"deterministic truncation, n = {N_KEEP}")
        ax_F.plot(M_VALUES, F_mean, "-", color="C2", label=f"qDRIFT-style, {n_draws} draws ({n_draws} runs)")
        ax_p.plot(M_VALUES, p_geo, "-", color="C2", label=f"qDRIFT-style, {n_draws} draws ({n_draws} runs)")
        ax_F.legend(fontsize=8)
    ax_F.axhline(1.0, color="k", ls="--", lw=0.8)
    ax_F.set_ylim(0, 1.05)
    ax_F.set_ylabel("fidelity with true ground state")
    ax_p.set_yscale("log"); ax_p.set_ylim(*p_lim)
    ax_p.set_ylabel(r"$p_{\rm succ}$")
    for ax in (ax_F, ax_p):
        ax.set_xlabel("polynomial order m")
    fig.suptitle(f"wall-Chebyshev, U = {U}, S = {'E0' if EXACT_S else 'HF'}: {title}", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"wallcheb_{name}.png", dpi=200)
plt.show()