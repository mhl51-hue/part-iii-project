'''
Fermi Hubbard model
Taking the second quantised hamiltonian and transforming it into a qubit hamiltonian
and then transforming into a matrix
'''

from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import (
    LineLattice,
    BoundaryCondition,
)
from qiskit_nature.second_q.problems import LatticeModelProblem
from qiskit_nature.second_q.mappers import JordanWignerMapper

# LineLattice initialises the space in which the fermi hubbard model exists
line_lattice = LineLattice(num_nodes=10, boundary_condition=BoundaryCondition.OPEN)

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
fermihubbard_op, aux_op = fermihubbard_problem.second_q_ops() 

# Using the Jordan Wigner transformation to get the qubit hamiltonian
mapper = JordanWignerMapper()
fermihubbard_qubit = mapper.map(fermihubbard_op)
