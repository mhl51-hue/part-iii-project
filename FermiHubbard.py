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
from dataclasses import dataclass

def fermihubbardmodel(U):
    '''
    Function on strength of onsite interactions U
    Returns H, the full fermi hubbard matrix
            num_electrons, electron counting matrix (number of electrons in a given quantum state)
    '''
    # LineLattice initialises the space in which the fermi hubbard model exists, 2 site system
    line_lattice = LineLattice(num_nodes=2, boundary_condition=BoundaryCondition.OPEN)

    '''
    Creating the fermi hubbard model
    t: interaction parameter, t = -1
    v: onsite potential, v = 0
    U: stength of onsite interaction, U = (1, 2, 4, 8)
    '''

    fermi_hubbard_model = FermiHubbardModel(
        line_lattice.uniform_parameters(
            uniform_interaction=-1,
            uniform_onsite_potential=0,
        ),
        onsite_interaction=U,
    )

    # Extracting the 2nd quantised hamiltonian from the fermi hubbard model
    fermihubbard_problem = LatticeModelProblem(fermi_hubbard_model)
    fermihubbard_op, aux_ops = fermihubbard_problem.second_q_ops() # Fermihubbar_op is the 2nd quantised H

    # Fermihubbard_op --JW mapping--> qubit hamiltonian --> matrix form --> array
    mapper = JordanWignerMapper()
    H = mapper.map(fermihubbard_op).to_matrix(sparse=True).toarray()

    return H

def half_filling_regime(H, tol=1e-9):
    '''
    Current form of H is a 16x16 matrix, function will slice matrix down to 4x4
    m_s = 0
    '''

    H = np.asarray(H)
    idx = np.arange(16)
    bit = lambda j: (idx >> j) & 1
    for up in ((0, 1), (2, 3), (0, 2), (1, 3)):
        dn = tuple(sorted(set(range(4)) - set(up)))
        N_up = np.diag(sum(bit(j) for j in up).astype(float))
        N_dn = np.diag(sum(bit(j) for j in dn).astype(float))
        if np.linalg.norm(H @ N_up - N_up @ H) < tol and np.linalg.norm(H @ N_dn - N_dn @ H) < tol:
            break
    else:
        raise ValueError("check line lattice size")

    indices = [i for i in range(16)
               if sum((i >> j) & 1 for j in up) == 1 and sum((i >> j) & 1 for j in dn) == 1]
    Hs = np.real(H[np.ix_(indices, indices)])
    # Neel determinants have no double occupancy -> diagonal energy = 0
    neel = [k for k in range(len(indices)) if abs(Hs[k, k]) < tol]
    if Hs.shape != (4, 4,) or not neel:
        raise ValueError("worng matrix structure")

    return Hs, neel[0], indices

# creating class for the fermi hubbard model
@dataclass
class FermiHubbard_class:
    U: float
    H: np.ndarray
    E: np.ndarray
    V: np.ndarray
    neel: int
    indices: int

    @property
    def E0(self):
        # ground state energy
        return self.E[0]

    @property
    def E1(self):
        # first excited state
        return self.E[self.E > self.E0 + 1e-8][0]

    @property
    def gap(self):
        return self.E1 - self.E0

# Calling the model and the slicing function to create the fermihubbard 4x4 matrix
def fermihubbard_subspace(U):
    H_subspace = fermihubbardmodel(U)
    H, neel, indices = half_filling_regime(H_subspace)
    E, V = np.linalg.eigh(H)
    return FermiHubbard_class(U=U, H=H, E=E, V=V, neel=neel, indices=indices)

# Defining exact theoretical groundstate energy
def analytic_E0(U):
    return U / 2 - np.sqrt(U ** 2 / 4 + 4 * -1 **2)