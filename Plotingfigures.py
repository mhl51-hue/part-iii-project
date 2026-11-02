
import numpy as np
import matplotlib.pyplot as plt

from FermiHubbard import fermihubbard_subspace, analytic_E0
from ApplyingProjector import evaluate_orders, projectors

U_vals = (1, 2, 4, 8)
colours = {1: "hotpink", 2: "blue", 4: "orange", 8: "purple"}

def run_all(U_vals, ground_state_energy, max_order=150):
    # Running the values of U and storing in a dictionary
    return {U: evaluate_orders(fermihubbard_subspace(U), ground_state_energy, max_order)
            for U in U_vals}

def plot_figure(results, fname, caption):
    # Plots two sets of graphs, one with energy and the other with fidelity 
    fig, axs = plt.subplots(len(projectors), 2, figsize=(11,10), sharex='col')
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (U, res) in enumerate(results.items()):
        colour = colours.get(U, cycle[i % len(cycle)])

        for row, projector in enumerate(projectors):
                    d = res["projectors"][projector]
                    axs[row, 0].plot(d["orders"], d["energy"], color=colour, lw=1.1, label=f"U = {U}")
                    axs[row, 0].axhline(res["E0"], color="k", ls="--", lw=0.8)     # exact ground state energy
                    axs[row, 1].plot(d["orders"], d["fidelity"], color=colour, lw=1.1, label=f"U = {U}")

    for row, projector in enumerate(projectors):
        axs[row, 0].set_title(projector, fontsize=10)
        axs[row, 1].set_title(projector, fontsize=10)
        axs[row, 0].set_ylabel("E")
        axs[row, 1].set_ylabel(r"$\langle\Psi_0|\Psi\rangle$")
        axs[row, 1].axhline(1.0, color="k", ls="--", lw=0.8)

    axs[0, 0].legend(fontsize=8, ncol=2)
    axs[-1, 0].set_xlabel("Polynomial order")
    axs[-1, 1].set_xlabel("Polynomial order")
    fig.suptitle(caption, fontsize=10)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    return fig

if __name__ == "__main__":
     fig4 = run_all(U_vals, ground_state_energy=True, max_order=150)
     plot_figure(fig4, "fig4.png", "Figure 4: ground state energy known a priori (S = E0)")

     fig5 = run_all(U_vals, ground_state_energy=False, max_order=150)
     plot_figure(fig5, "fig5.png", "Figure 5: ground state energy known a priori (S = E0)")

     plt.show()