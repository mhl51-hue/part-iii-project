'''
The distribution the samplers draw from: |coefficient| of every non-identity Pauli string, largest first, for each U.
Bars are coloured by type; the number above each bar is that string's sampling probability p_j = |h_j| / alpha.
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
 
from FermiHubbard import fermihubbard_subspace
from DeterministicSampling import pauli_terms
 
u_vals = (1, 2, 4, 8)
fig, axs = plt.subplots(1, len(u_vals), figsize=(14, 3.8), sharey=True)
for ax, U in zip(axs, u_vals):
    terms = sorted(pauli_terms(fermihubbard_subspace(U).qubit_H)[1], key=lambda t: -abs(t[1]))
    w = np.array([abs(c) for _, c in terms])
    hopping = [("X" in l or "Y" in l) for l, _ in terms]
    ax.bar(range(len(w)), w, color=["C0" if h else "C1" for h in hopping])
    for i, x in enumerate(w / w.sum()):
        ax.text(i, w[i] + 0.03, f"{x:.2f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(w)), [l for l, _ in terms], rotation=90, fontsize=7)
    ax.set_title(f"U = {U}", fontsize=9)
axs[0].set_ylabel(r"$|h_j|$")
axs[0].legend(handles=[Patch(color="C0", label="hopping (XX, YY)"), Patch(color="C1", label="on-site (Z, ZZ)")], fontsize=8)
fig.suptitle("Pauli coefficients in decreasing order; labels = sampling probability p_j", fontsize=10)
fig.tight_layout()
fig.savefig("pauli coefficients.png", dpi=200)
plt.show()
