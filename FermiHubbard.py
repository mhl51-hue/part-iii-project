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
    fermihubbard_op, _ = fermihubbard_problem.second_q_ops() # Fermihubbar_op is the 2nd quantised H

    # Fermihubbard_op --JW mapping--> qubit hamiltonian --> matrix form --> array
    mapper = JordanWignerMapper()
    qubit_H = mapper.map(fermihubbard_op)
    H_full = qubit_H.to_matrix(sparse=True).toarray()

    return qubit_H, H_full

def half_filling_indices(H, tol=1e-9):
    '''
    Indices of the 16x16 matrix
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

    return [i for i in range(16)
               if sum((i >> j) & 1 for j in up) == 1 and sum((i >> j) & 1 for j in dn) == 1]
    
def half_filling_regime(H, tol=1e-9):
    '''
    Slicing the 16x16 matrix down to a 4x4
    '''
    keep = half_filling_indices(H, tol)
    H = np.real(H[np.ix_(keep, keep)])

    # Neel determinants have no double occupancy -> diagonal energy = 0
    neel = [k for k in range(4) if abs(H[k, k]) < tol]

    if not neel:
        raise ValueError("non zero energy det in subspace")

    return H, neel[0]

# creating class for the fermi hubbard model
@dataclass
class FermiHubbard_class:
    U: float
    qubit_H: object # used for alpha value
    H_full: np.ndarray # full 16x16 matrix
    H: np.ndarray # half filled 4x4 Hamiltonian
    E: np.ndarray
    V: np.ndarray
    neel: int

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
    qubit_H, H_full = fermihubbardmodel(U)
    H, neel = half_filling_regime(H_full)
    E, V = np.linalg.eigh(H)
    return FermiHubbard_class(U=U, qubit_H=qubit_H, H_full=H_full, H=H, E=E, V=V, neel=neel)

# Defining exact theoretical groundstate energy
def analytic_E0(U):
    return U / 2 - np.sqrt(U ** 2 / 4 + 4 * -1 **2)