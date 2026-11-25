'''
Figure 3: theoretical success probability
Product: product of LCUs, eq. 36
GQSVT 1: P1, direct encoding, eq. 40
GQSVT 2: P2, rescaled nodes, eq. 42
'''
import numpy as np
import matplotlib.pyplot as plt
from SuccessProbability import p_succ_wallcheb_product_approx, p_succ_wallcheb_P1_approx, p_succ_wallcheb_P2_approx

colours = {2: 'blue', 4: 'orange', 6: 'lightblue', 8: 'green', 10: 'yellow'}
a = np.linspace(1, 3, 21)

panels = [
    ("Product", lambda m: p_succ_wallcheb_product_approx(a, m)),
    ("GQSVT P1", lambda m: p_succ_wallcheb_P1_approx(a, m)),
    ("GQSVT P2", lambda m: np.array([p_succ_wallcheb_P2_approx(ai, m) for ai in a])),
]

fig, axs = plt.subplots(1, 3, figsize=(9, 4.6), sharey=True)
for ax, (title, p_over_c0sq) in zip(axs, panels):
    for m in (2, 4, 6, 8, 10):
        ax.plot(a, p_over_c0sq(m), color=colours[m], lw=1.2, label=f"m = {m}")
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 1)
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel(r"$\alpha/||H||$")
    ax.set_title(title, fontsize=10)
axs[0].set_ylabel(r"$p_{\rm succ}/c_0^2$")
axs[0].legend(fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig("fig3.png", dpi=220)
plt.show()
