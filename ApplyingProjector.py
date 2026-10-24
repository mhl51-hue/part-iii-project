import numpy as np
from numpy.polynomial import chebyshev as Cheb
from FermiHubbard import fermihubbard_matrix
from HartreeFock import psi_HF
from Projectors import (
    wall_cheb,
    delta_func,
    step_func,
)

'''
Rescaling the fermi hubbard matrix
'''
H = fermihubbard_matrix.toarray()
evals, evecs = np.linalg.eigh(H)

E0 = evals[0]
E1 = evals[evals > E0 + 1e-8][0] # First distinct excited level
Emax = evals[-1]

R = 1.01 * (Emax - E0)
M = (H - E0 * np.eye(H.shape[0])) / R
Scaled_gap = (E1 - E0) / R # Scaled energy gap

# Single value decomposition SVD, M = U diag(s) V†
U, s, Vt = np.linalg.svd(M)
V = Vt.conj().T

'''
Applying the polynomial functions to the single value matrix
1. Wall-Chebyshev
2. Eigenstate filter
3. Step function
'''
# exact ground-space projector, for comparison
gs = np.abs(evals - E0) < 1e-8
P_exact = evecs[:, gs] @ evecs[:, gs].conj().T

m = int(np.ceil(2 / Scaled_gap))
p_wallcheb = wall_cheb(s * R, 0.0, R, m) # Wall-Chebyshev projector evaluated at each s[i]

# Eigenstate filter (delta function)
l = int(np.ceil(3 / Scaled_gap))
p_delta = delta_func(s, l, Scaled_gap) # Delta function evaluated at each s[i]

# Step function
coefs_step, k = step_func(deg=int(5 * np.sqrt(2) / Scaled_gap), delta = Scaled_gap, mu = Scaled_gap / 2)
p_step = Cheb.chebval(s, coefs_step) # Step function evaluated at each s[i]


'''
Applying the polynomials to singular values
P = V diag(p(s)) V†
'''

def apply_projector(p_s):
    return V @ np.diag(p_s) @ V.conj().T

projectors = {
    "wall_Chebyshev": p_wallcheb,
    "eigenstate filter": p_delta,
    "step function": p_step,
}

for name, p_s in projectors.items():
    P = apply_projector(p_s)
    phi = P @ psi_HF # applying the projectors onto the HF state
    success_prob = np.vdot(phi, phi).real
    phi = phi / np.linalg.norm(phi)
    energy = np.vdot(phi, H @ phi).real
    overlap = np.vdot(phi, P_exact @ phi).real
    print(f"{name:18s} success prob = {success_prob:.4f}  "
          f"E = {energy:.6f} overlap after = {overlap:.6f}")

'''
something is wrong, success prob = 0?
'''

