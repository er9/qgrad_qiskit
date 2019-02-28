"""Script performing an example variational algorithm using Qiskit Aqua."""

# =======
# imports
# =======

import numpy as np
import matplotlib.pyplot as plt

from qiskit import Aer, IBMQ, execute
from qiskit.providers.aer import noise
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor

from qiskit_aqua.components.initial_states import Zero
from qiskit_aqua.components.variational_forms import RY
from qiskit_aqua.operator import Operator
from qiskit_aqua.components.optimizers import COBYLA
from qiskit_aqua.algorithms import VQE
from qiskit_aqua import QuantumInstance

import finite_diff 
import aqgd



# =========
# constants
# =========

# number of qubits
num_qubits = 2

# pauli operators for a hamiltonian
pauli_dict = {
            'paulis': [{"coeff": {"imag": 0.0, "real": -1.052373245772859}, "label": "II"},
                       {"coeff": {"imag": 0.0, "real": 0.39793742484318045}, "label": "IZ"},
                       {"coeff": {"imag": 0.0, "real": -0.39793742484318045}, "label": "ZI"},
                       {"coeff": {"imag": 0.0, "real": -0.01128010425623538}, "label": "ZZ"},
                       {"coeff": {"imag": 0.0, "real": 0.18093119978423156}, "label": "XX"}
                       ]
    }

# ======================
# setting up the circuit
# ======================

# define the initial state
init_state = Zero(num_qubits)

# get a variational ansatz
ansatz = RY(num_qubits) #,initial_state=init_state)

# operator from hamiltonian
qubit_op = Operator.load_from_dict(pauli_dict)

# get an optimizer
# optim = COBYLA(maxiter=1000, disp=True)
optim   = aqgd.AQGD(maxiter=25, disp=True, eta=1.0, phase_noise=1./2)
# optim   = finite_diff.Finite_Diff(maxiter=15,disp=True,eta=1.0)

# form the algorithm
vqe = VQE(qubit_op, ansatz, optim)
# print('---------------------- \n')
# exit()
# ansatz = RY(num_qubits,initial_state=init_state)
# vqe = VQE(qubit_op, ansatz, optim)

# get a backend
backend = Aer.get_backend("statevector_simulator")

# get a quantum instance
qinstance = QuantumInstance(backend, shots=1024)

# ===================
# do the optimization
# ===================

result = vqe.run(qinstance)

# ================
# show the results
# ================

# output of the optimization
print(result)

# show the circuit
circuit = vqe.construct_circuit(list(range(8)), backend)[0]
print(circuit)
