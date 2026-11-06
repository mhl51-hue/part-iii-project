'''
Success Probability

Calculates the projection outcomes on the state as well as the alpha value from LCU block encoding
Calculates the success probabilitly of each projector
'''

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import chebyshev as Cheb
from scipy.special import erf
from Projectors import (
    wall_cheb_coefs, 
    delta_func,
    step_func,
    step,
    )
from ApplyingProjector import estimate_energy_and_range

'''
Calculating alpha value
1-norm of the Pauli LCU decomposition
Appendix B
'''
def projection_outcome(E, c, w):
    '''
    Apply weights w_i = f(E_i) to a state with overlaps c_i on eigenstate
    returns p: squared norm of psi, success probability
            energy: averaged energy of normalised state
            fidelity: overlap of the state after projector has acted on it with the true ground state
        
    The weights of the projectors are run through this function and this returns the success probability
    '''
    prob = (np.asarray(c) * np.asarray(w)) ** 2
    p = prob.sum()
    if not np.isfinite(p) or p < 1e-300:
        return dict(p=p, energy=np.nan, fidelity=np.nan)
    return dict(p=p, energy=float((prob * E).sum() / p), fidelity=float(np.sqrt(prob[0] / p)))

def lcu_alpha(qubit_H, scale=1.0, shift=0.0):
    '''
    returns alpha, the sub normalisation of the LCU block encoding

    '''
    h_id = rest = 0
    for label, coef in qubit_H.simplify().to_list():
        if set(label) == {"I"}:
            h_id += coef.real
        else:
            rest += abs(coef)
    return abs(scale) * rest + abs(scale * h_id + shift)

'''
Calculating the success probabilities of the projectors
1. wall-chebyshev
2. delta function
3. step function
'''

# wall-chebyshev GQSVT encoding (eqs. 39-40)
def p_succ_wallcheb(E, c, S, R, alpha, m):
    x = 2 * (np.asarray(E) - S) / R - 1 # eigenvalues of H_tilda
    return projection_outcome(E, c, Cheb.chebval(x / alpha, wall_cheb_coefs(m)))

def p_succ_wallcheb_approx(alpha, m):
    # Can only use for S=E0
    return Cheb.chebval(-1.0 / np.asarray(alpha, dtype=float), wall_cheb_coefs(m)) ** 2

# eigenstate filter (eqs 9., E16-19)
def p_succ_delta(E, c, S, alpha_delta, l, gap):
    x = (np.asarray(E) - S) / alpha_delta
    return projection_outcome(E, c, delta_func(x, l, gap / alpha_delta))

# step function (eqs. E20-25)
def p_succ_step(E, c, S, alpha_step, gap, deg=50, epsilon=1e-2):
    g = gap / alpha_step
    coefs, _ = step_func(deg=deg, delta=g / 4, mu=g/2, epsilon=epsilon)
    coefs /= max(1.0, np.max(np.abs(step(np.linspace(-1, 1, 4001), coefs))))
    return projection_outcome(E, c, step((np.asarray(E) - S) / alpha_step, coefs))

# gathering results from all the projectors for one value of U
def success_results(subspace, psi0, ground_state_energy, m=10):
    S, R = estimate_energy_and_range(subspace, ground_state_energy)
    c = subspace.V.T @ psi0
    mu = S + subspace.gap/2

    alpha_wall = lcu_alpha(subspace.qubit_H, 2/R, -(2*S/R+1))
    norm = np.max(np.abs(2*(np.linalg.eigvalsh(subspace.H_full)-S)/R-1))
    assert alpha_wall >= norm - 1e-9, f"alpha = {alpha_wall:.3f} < ||H-|| = {norm:.3f}: not a valid block encoding"

    return dict(
        S=S, R=R, c0_sq=c[0] ** 2, alpha_H=lcu_alpha(subspace.qubit_H), alpha_wall=lcu_alpha(subspace.qubit_H, 2/R, -(2*S/R+1)), norm_wall = norm,
        wall=p_succ_wallcheb(subspace.E, c, S, R, lcu_alpha(subspace.qubit_H, 2/R, -(2*S/R+1)), m),
        filter=p_succ_delta(subspace.E, c, S, lcu_alpha(subspace.qubit_H, shift=-S), m // 2, subspace.gap),
        step=p_succ_step(subspace.E, c, mu, lcu_alpha(subspace.qubit_H, shift=-mu), subspace.gap),
    )

if __name__ == "__main__":
    from FermiHubbard import fermihubbard_subspace
    from HartreeFock import hartree_fock_state

    m=10
    print(f"Success probability of the projectors, all with m={m}")
    for U in (1, 2, 4, 8):
        subspace = fermihubbard_subspace(U)
        psi0, _ = hartree_fock_state(subspace)
        for exact in (True, False):
            r = success_results(subspace, psi0, exact, m)
            shortcut = f"{r['c0_sq'] * p_succ_wallcheb_approx(r['alpha_wall'], m):.2e}" if exact else "n/a"
            print(f"U={U} S={'E0' if exact else 'HF'}:"
                  f"Wall-chebyshev(QSVT1):{r['wall']['p']:.2e} [wallcheb success approx: {shortcut}]  "
                  f"eigenstate filter:{r['filter']['p']:.3f}  step function:{r['step']['p']:.3f}")
