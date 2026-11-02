'''
Applying the projectors to the fermihubbard matrix
and then acting that on the Hartree fock state vector
'''

import numpy as np
from numpy.polynomial import chebyshev as Cheb
from HartreeFock import hartree_fock_state
from Projectors import (
    wall_cheb,
    delta_func,
    step_func,
    step
)

projectors = ("Eigenstate filter", "Step function", "wall-Chebyshev")

def estimate_energy_and_range(subspace, ground_state_energy, beta=1.1):
    '''
    S: estimate of the ground state energy
    R: estimate spectral range R = beta * E_N-1 - s (eq. 21), where E_N-1 is the Gershgorin
    estimate of the highest eigenvalue (eq. 20): diagonal element plus sum of off-diagonals
    '''
    H = subspace.H
    gersh = max(H[i,i] + np.sum(np.abs(H[i])) - abs(H[i,i]) for i in range(len(H)))
    S = subspace.E0 if ground_state_energy else H[subspace.neel, subspace.neel]
    return S, beta * gersh - S

def projector_weighted(projector, E, S, R, order, gap):
    '''
    Scales the projectors using R and S
    '''
    gap_scaled = gap / R
    x = (E - S) / R
    if projector == "wall-Chebyshev":
        return wall_cheb(E, S, R, order)
    if projector == "Eigenstate filter":
        return delta_func(x, order // 2, gap_scaled)
    if projector == "Step function":
        coefs, _ = step_func(order, delta=gap_scaled / 4, mu=gap_scaled / 2)
        return step(x, coefs)
    raise ValueError(projector)

def apply_projector(subspace, filter_E, psi0):
    '''
    P = V diag(f(E)) V^T, applied to psi0 and renormalised
    Returns (energy, fidelity with exact ground state)
    filter_E, filter functions evaluated at energies E
    '''
    P = subspace.V @ np.diag(filter_E) @ subspace.V.T
    phi = P @ psi0
    norm = np.linalg.norm(phi)
    if not np.isfinite(norm) or norm < 1e-300: # projector anihilated the state
        return np.nan, np.nan
    phi = phi / norm
    energy = phi @ subspace.H @ phi
    fidelity = abs(subspace.V[:, 0] @ phi)
    return energy, fidelity

def evaluate_orders(subspace, ground_state_energy, max_order=150, beta=1.1):
    '''
    Apply the projectors to the HF state for differing polynomial orders
    The filter and step func are even in x, so only even orders are meaningful for them
    '''
    psi0, _ = hartree_fock_state(subspace)
    S, R = estimate_energy_and_range(subspace, ground_state_energy, beta)

    results = {}
    for projector in projectors:
        step_size = 2 if projector in ("Eigenstate filter", "Step function") else 1
        orders = np.arange(0, max_order + 1, step_size)
        out = np.array([apply_projector(subspace,
                                        projector_weighted(projector, subspace.E, S, R, int(m), subspace.gap),
                                        psi0)
                                    for m in orders])
        results[projector] = dict(orders=orders, energy=out[:, 0], fidelity=out[:, 1])
    return dict(U=subspace.U, S=S, R=R, E0=subspace.E0, projectors=results)

'''
Checking projector ability for U=1
prints for initial state of known E0 and for HF energy
'''
if __name__ == "__main__":
    from FermiHubbard import fermihubbard_subspace
    subspace = fermihubbard_subspace(1)
    for exact in (True, False):
        r = evaluate_orders(subspace, exact, max_order=60)
        print(f"U=1, S {'=E0' if exact else '= HF energy'}: S = {r['S']:.3f}, R = {r['R']:.3f}")
        for m, d in r["projectors"].items():
            print(f"   {m:18s} final E = {d['energy'][-1]: .4f}   fidelity = {d['fidelity'][-1]:.4f}")