'''
Setting the Hartree Fock state
Using n = 2 to match the Fermi hubbard model set up

Produces psi_HF which is the state vector
'''

from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit import QuantumCircuit
from Older_code_model.FermiHubbard_rough import H
import numpy as np

# Creating the Hartree Fock occupation state
hartreefock_circuit = HartreeFock(
    num_spatial_orbitals=2,
    num_particles=(1,1), # n_up, n_down
    qubit_mapper=JordanWignerMapper(),
)

psi_HF = Statevector(hartreefock_circuit).data

# Checking the statevector
if __name__ == "__main__":
    a = np.argmax(np.abs(psi_HF))
    print("basis index:", a, "bits:", format(a, "08b"))

'''
Checking number of electrons in HF state and fermi hubbard state
'''

n_sites = 2
idx = np.arange(2 ** (2*n_sites))
n_up = np.array([bin(i & (2**n_sites - 1)).count("1") for i in idx])
n_dn = np.array([bin(i >> n_sites).count("1") for i in idx])

evals, evecs = np.linalg.eigh(H)
if __name__ == "__main__":
    for name, v in [("HF state", psi_HF), ("ground state", evecs[:,0])]:
        p = np.abs(v) ** 2
        print(f"{name:13s} <N_up> = {np.sum(p * n_up):.3f} <N_dn> = {np.sum(p * n_dn):.3f}")