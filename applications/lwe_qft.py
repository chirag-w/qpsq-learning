import numpy as np
from numpy import pi

from qiskit import QuantumCircuit
import qiskit.quantum_info as qi

import sys

sys.path.append('../')

from predict import *
from states import *
from obs import *

#Number of qudits in input
n = 5

#Dimension of qudit
q = 2

#Number of qubits in a qudit of dimension q
num_qubits = int(np.log2(q))
assert q == 2**num_qubits

#QFT applied to each qudit
U = [1]
for i in range(n+1):
    U = np.kron(U, qft_unitary(q))
# U = np.eye(2**n)

O = pauli_one_obs((n+1)*num_qubits,(n+1)*num_qubits-1,'Z')

#Target state distribution
D = []
size_D = 100

#LWE states
D = lwe_samples(size_D, s = '10101', n = n, q = q , error_distr = [2/3,1/3])

step = 200
N_min = 200
N_max = 2000
eps = 0.9



for N in range(N_min,N_max+1,step):
    x = learn(U,O,N,(n+1)*num_qubits,eps)
    # print(x)
    error = 0.0
    count = 0
    for rho in D:
        pred_val = pred(x,rho)
        # print('Predicted value:',pred_val)

        true_val = true_value(O,U,rho)
        # print('True value:',true_val)

        if pred_val * true_val > 0:
            count+= 1
        error += (true_val-pred_val)**2
    error /= size_D
    
    print("N = ",N,", error = ",error)
    print(count)