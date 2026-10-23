'''
Fermi Hubbard model
Taking the second quantised hamiltonian and transforming it into a qubit hamiltonian
and then transforming into a matrix
'''
import numpy as np
from numpy.polynomial import chebyshev as Cheb

from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import (
    LineLattice,
    BoundaryCondition,
)
from qiskit_nature.second_q.problems import LatticeModelProblem
from qiskit_nature.second_q.mappers import JordanWignerMapper


# LineLattice initialises the space in which the fermi hubbard model exists
line_lattice = LineLattice(num_nodes=4, boundary_condition=BoundaryCondition.OPEN)
# Need to check system size issues

'''
Creating the fermi hubbard model
t = -1.0 is the interaction parameter
v = 0.0 is the onsite potential
U = 5.0 is stength of onsite interaction
'''
fermi_hubbard_model = FermiHubbardModel(
    line_lattice.uniform_parameters(
        uniform_interaction=-1.0,
        uniform_onsite_potential=0.0,
    ),
    onsite_interaction=5.0,
)

# Extracting the 2nd quantised hamiltonian from the fermi hubbard model
fermihubbard_problem = LatticeModelProblem(fermi_hubbard_model)
fermihubbard_op, aux_op = fermihubbard_problem.second_q_ops() # Fermihubbar_op is the 2nd quantised H

# Using the Jordan Wigner transformation to get the qubit hamiltonian
mapper = JordanWignerMapper()
fermihubbard_qubit = mapper.map(fermihubbard_op) # Fermihubbard_qubit is the qubit H

# Turning the qubit hamiltonian into matrix form
fermihubbard_matrix = fermihubbard_qubit.to_matrix(sparse=True)

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

from Projectors import (
    wall_cheb,
    delta_func,
    step_func,
)

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
