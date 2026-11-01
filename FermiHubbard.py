'''
Fermi Hubbard model
Taking the second quantised hamiltonian and transforming it into a qubit hamiltonian
and then transforming into a matrix

Produces fermihubbard_matrix
'''
import numpy as np
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import (
    LineLattice,
    BoundaryCondition,
)
from qiskit_nature.second_q.problems import LatticeModelProblem
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp


# LineLattice initialises the space in which the fermi hubbard model exists, 2 site system
line_lattice = LineLattice(num_nodes=2, boundary_condition=BoundaryCondition.OPEN)

'''
Creating the fermi hubbard model
t: interaction parameter
v: onsite potential
U: stength of onsite interaction, varying U for figures 4,5
'''

fermi_hubbard_model = FermiHubbardModel(
    line_lattice.uniform_parameters(
        uniform_interaction=-1.0,
        uniform_onsite_potential=0,
    ),
    onsite_interaction=2.0,
)

# Extracting the 2nd quantised hamiltonian from the fermi hubbard model
fermihubbard_problem = LatticeModelProblem(fermi_hubbard_model)
fermihubbard_op, aux_ops = fermihubbard_problem.second_q_ops() # Fermihubbar_op is the 2nd quantised H

'''
2 site, 2 spin system
Ensuring half filling
mode 0: site 0, spin up
mode 1: site 0, spin down
mode 2: site 1, spin up
mode 3: site 1, spin down
'''

num_modes = 4
num_particles_terms = {}
for mode in range(4):
    label = f"+_{mode} -_{mode}"
    num_particles_terms[label] = 1.0

# Total particle counter, counts how many electrons are present in a given quantum state
num_electrons_op = FermionicOp(num_particles_terms, num_spin_orbitals=num_modes)

# Using the Jordan Wigner transformation to get the qubit hamiltonian
mapper = JordanWignerMapper()

# Fermihubbard_op --JW mapping--> qubit hamiltonian --> matrix form --> array
H = mapper.map(fermihubbard_op).to_matrix(sparse=True).toarray()
num_electrons = mapper.map(num_electrons_op).to_matrix(sparse=True).toarray() # Electron counting operator in matrix form
evals, evecs = np.linalg.eigh(H)

# Filtering the eigenvalues to ensure half filling
half_filled_evals = np.array([
    evals[i] for i in range(len(evals))
    if np.isclose(np.real(evecs[:, i].conj().T @ num_electrons @ evecs[:, i]), 2.0)
])

E0 = half_filled_evals[0]
E1 = half_filled_evals[half_filled_evals > E0 + 1e-8][0] # First distinct excited level
Emax = half_filled_evals[-1]

if __name__ == "__main__":
    print(f"Ground state energy: {E0:.4f}")