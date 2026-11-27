'''
Success Probability

Calculates the projection outcomes on the state as well as the alpha value from LCU block encoding
Calculates the success probabilitly of each projector

wall-Chebyshev encodings (section 4 of the paper):
    Product: product of LCUs, eqs. 34-36
    P1: GQSVT direct encoding, eqs. 39-40
    P2: GQSVT with rescaled nodes, eqs. 41-42
'''

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
from FermiHubbard import fermihubbard_subspace
from HartreeFock import hartree_fock_state

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
            fidelity: squared overlap of the state after projector has acted on it with the true ground state
        
    The weights of the projectors are run through this function and this returns the success probability
    '''
    prob = (np.asarray(c) * np.asarray(w)) ** 2
    p = prob.sum()
    if not np.isfinite(p) or p < 1e-300:
        return dict(p=p, energy=np.nan, fidelity=np.nan)
    return dict(p=p, energy=float((prob * E).sum() / p), fidelity=float(prob[0] / p))

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
1. Wall-chebyshev success probability
i. product form
ii. P1
iii. P2
'''

def wallcheb_nodes_scaled(m):
    '''
    nodes of the order m wall-Chebyshev polynomial in the rescaled variable x (eigenvalues of H~)
    a~_v = -cos(v pi / (m + 1/2)), v = 1..m  (eq. 16 mapped through eq. 14)
    '''
    return -np.cos(np.arange(1, m + 1) * np.pi / (m + 0.5))

# i. wall-Chebyshev product of LCUs (eqs. 34-36)
def product_alphas(qubit_H, S, R, m):
    '''
    alpha_v of every factor in the product form
    factor v is H~ - a~_v = (2/R) H - (2S/R + 1 + a~_v) I, block encoded on its own with LCU,
    so alpha_v is the LCU 1-norm of that operator (identity coefficient includes the shift a~_v)
    '''
    return np.array([lcu_alpha(qubit_H, 2/R, -(2*S/R + 1 + a)) for a in wallcheb_nodes_scaled(m)])

def wallcheb_product(x, alphas):
    '''
    PRODUCT form weight on an eigenvalue x of H~: prod_v (x - a~_v) / alpha_v
    alphas: the m values alpha_v from product_alphas
    '''
    nodes = wallcheb_nodes_scaled(len(alphas))
    return np.prod((np.asarray(x, dtype=float)[..., None] - nodes) / np.asarray(alphas), axis=-1)

def p_succ_wallcheb_product(E, c, S, R, alphas):
    '''PRODUCT form success probability, eq. 35'''
    x = 2 * (np.asarray(E) - S) / R - 1
    return projection_outcome(E, c, wallcheb_product(x, alphas))

def p_succ_wallcheb_product_approx(alpha, m):
    '''
    Eq. 36 shortcut p / c0^2 = prod_v (E0 - a_v)^2 / alpha_v^2. Only for S = E0 and large m.
    Uses the upper bound alpha_v = alpha + |a~_v| (as in the Product panel of Figure 3), alpha is the alpha of H~.
    '''
    nodes = wallcheb_nodes_scaled(m)
    alpha = np.asarray(alpha, dtype=float)[..., None]
    return np.prod((-1 - nodes) ** 2 / (alpha + np.abs(nodes)) ** 2, axis=-1)

# ii. wall-Chebyshev P1, GQSVT direct encoding (eqs. 39-40)
def p_succ_wallcheb_P1(E, c, S, R, alpha, m):
    x = 2 * (np.asarray(E) - S) / R - 1 # eigenvalues of H_tilda
    return projection_outcome(E, c, Cheb.chebval(x / alpha, wall_cheb_coefs(m)))

def p_succ_wallcheb_P1_approx(alpha, m):
    # P1 approx, eq. 40. Can only use for S=E0
    return Cheb.chebval(-1.0 / np.asarray(alpha, dtype=float), wall_cheb_coefs(m)) ** 2

# iii. wall-Chebyshev P2, GQSVT with rescaled nodes (eqs. 41-42)
def wallcheb_P2(y, alpha, m):
    '''
    P2(y) = prod_v (y - a_v / alpha),  y = H~/alpha,  nodes a_v = -cos(v pi / (m + 1/2)).
    Eq. 41 does not say how it is scaled; QSVT needs |P2| <= 1, so it is divided by its maximum on [-1, 1].
    That choice reproduces the GQSVT 2 panel of Figure 3 (all curves start at 1 for alpha = 1).
    '''
    roots = -np.cos(np.arange(1, m + 1) * np.pi / (m + 0.5)) / alpha
    prod = lambda z: np.prod(np.asarray(z, dtype=float)[..., None] - roots, axis=-1)
    return prod(y) / np.max(np.abs(prod(np.linspace(-1, 1, 2001))))

def p_succ_wallcheb_P2(E, c, S, R, alpha, m):
    x = 2 * (np.asarray(E) - S) / R - 1
    return projection_outcome(E, c, wallcheb_P2(x / alpha, alpha, m))
 
 
def p_succ_wallcheb_P2_approx(alpha, m):
    # P2 approx, eq. 40. Can only use for S=E0
    return float(wallcheb_P2(-1.0 / alpha, alpha, m)) ** 2


'''
2. Delta function 
(eqs 9., E16-19)
'''
def p_succ_delta(E, c, S, alpha_delta, l, gap):
    x = (np.asarray(E) - S) / alpha_delta
    return projection_outcome(E, c, delta_func(x, l, gap / alpha_delta))

'''
3. Step function 
(eqs. E20-25)
'''
def p_succ_step(E, c, S, alpha_step, gap, deg=500, epsilon=1e-2):
    g = gap / alpha_step
    coefs, _ = step_func(deg=deg, delta=g / 4, mu=g/2, epsilon=epsilon)
    coefs /= max(1.0, np.max(np.abs(step(np.linspace(-1, 1, 4001), coefs))))
    return projection_outcome(E, c, step((np.asarray(E) - S) / alpha_step, coefs))

# gathering results from all the projectors for one value of U
def success_results(subspace, psi0, ground_state_energy, m=10):
    S, R = estimate_energy_and_range(subspace, ground_state_energy)
    c = subspace.V.T @ psi0

    alpha_wall = lcu_alpha(subspace.qubit_H, 2/R, -(2*S/R+1))
    norm = np.max(np.abs(2*(np.linalg.eigvalsh(subspace.H_full)-S)/R-1))
    assert alpha_wall >= norm - 1e-9, f"alpha = {alpha_wall:.3f} < ||H-|| = {norm:.3f}: not a valid block encoding"

    alphas = product_alphas(subspace.qubit_H, S, R, m)

    return dict(
        S=S, R=R, c0_sq=c[0] ** 2, alpha_H=lcu_alpha(subspace.qubit_H), alpha_wall=alpha_wall, norm_wall = norm,
        alpha_product=alphas,
        wall_product=p_succ_wallcheb_product(subspace.E, c, S, R, alphas),   
        wall_P1=p_succ_wallcheb_P1(subspace.E, c, S, R, alpha_wall, m),          
        wall_P2=p_succ_wallcheb_P2(subspace.E, c, S, R, alpha_wall, m),
        filter=p_succ_delta(subspace.E, c, S, lcu_alpha(subspace.qubit_H, shift=-S), m // 2, subspace.gap),
        step=p_succ_step(subspace.E, c, S, lcu_alpha(subspace.qubit_H, shift=-S), subspace.gap),
    )

if __name__ == "__main__":
    m=10
    print(f"Success probability of the projectors, all with m={m}")
    for U in (1, 2, 4, 8):
        subspace = fermihubbard_subspace(U)
        psi0, _ = hartree_fock_state(subspace)
        for exact in (True, False):
            r = success_results(subspace, psi0, exact, m)
            shortcut = f"{r['c0_sq'] * p_succ_wallcheb_product_approx(r['alpha_wall'], m):.2e}" if exact else "n/a"
            print(f"U={U} S={'E0' if exact else 'HF'}: "
                  f"wall-Chebyshev product:{r['wall_product']['p']:.2e} [eq. 36 shortcut, alpha_v upper bound: {shortcut}]  "
                  f"(comparison: P1 {r['wall_P1']['p']:.2e}, P2 {r['wall_P2']['p']:.2e})  "
                  f"eigenstate filter:{r['filter']['p']:.3f}  step function:{r['step']['p']:.3f}")

