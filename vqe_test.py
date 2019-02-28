"""Script performing an example variational algorithm using Qiskit Aqua."""

# =======
# imports
# =======

from qiskit_aqua import get_aer_backend

from qiskit_aqua.components.initial_states import Zero
from qiskit_aqua.components.variational_forms import RY
from qiskit_aqua.operator import Operator
from qiskit_aqua.components.optimizers import COBYLA
from qiskit_aqua.algorithms import VQE
from qiskit_aqua import QuantumInstance

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
ansatz = RY(num_qubits,initial_state=init_state)

# operator from hamiltonian
qubit_op = Operator.load_from_dict(pauli_dict)

# get an optimizer
optimizer = COBYLA(maxiter=1000, disp=True)

# form the algorithm
vqe = VQE(qubit_op, ansatz, optimizer)

# get a backend
backend = get_aer_backend("statevector_simulator")

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
