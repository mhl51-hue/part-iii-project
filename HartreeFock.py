'''
Setting the Hartree Fock state
Using n = 2 to match the Fermi hubbard model set up
'''

import numpy as np

# Creating HF state for the 4x4 subspace taken from the larger fermihubbard matrix
def hartree_fock_state(subspace):
    '''
    returns (psi, energy), the Neel determinant as a vector
    '''
    psi = np.zeros(len(subspace.H))
    psi[subspace.neel] = 1.0
    return psi, float(subspace.H[subspace.neel, subspace.neel])

# Printing HF energy + overlap with ground state for each value of U, HF E should be zero for all U
if __name__ == "__main__":
    from FermiHubbard import fermihubbard_4x4
    for U in (1, 2, 4, 8):
        s = fermihubbard_4x4(U)
        psi, e = hartree_fock_state(s)
        print(f"U={U}: HF energy = {e:.3f}, overlap with ground state = {abs(s.V[:, 0] @ psi):.3f}")