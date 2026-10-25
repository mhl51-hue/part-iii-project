'''
Setting the Hartree Fock state
Using n = 4 to match the Fermi hubbard model set up

Produces psi_HF which is the state vector
'''

from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit import QuantumCircuit
from FermiHubbard import fermihubbard_matrix
import numpy as np

# Creating the Hartree Fock occupation state
hartreefock_circuit = HartreeFock(
    num_spatial_orbitals=4,
    num_particles=(2,2), # n_up, n_down
    qubit_mapper=JordanWignerMapper(),
)

psi_HF = Statevector(hartreefock_circuit).data

# Checking the statevector
if __name__ == "__main__":
    a = np.argmax(np.abs(psi_HF))
    print("basis index:", a, "bits:", format(a, "08b"))
