import numpy as np
import matplotlib.pyplot as plt
from SuccessProbability import p_succ_wallcheb_approx

colours = {2: 'blue', 4: 'orange', 6: 'lightblue', 8: 'green', 10: 'yellow'}

fig, ax = plt.subplots(figsize=(3.2, 4.6))
a_coarse = np.linspace(1, 3, 21)
for m in (2, 4, 6, 8, 10):
    ax.plot(a_coarse, p_succ_wallcheb_approx(a_coarse, m), color=colours[m], lw=1.2, label=f"m = {m}")
ax.set_yscale("log")
ax.set_ylim(1e-8, 1)
ax.set_xticks([1, 2, 3])
ax.set_xlabel(r"$\alpha/||H||$")
ax.set_ylabel(r"$p_{\rm succ}/c_0^2$")
ax.set_title("GQSVT 1", fontsize=10)
ax.legend(fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig("fig3_qsvt1.png", dpi=220)
plt.show()