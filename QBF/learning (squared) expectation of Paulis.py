#from Chirag's github

import numpy.linalg as lg
import cmath

n = 3
num_qubits = n

Hadamard = np.array([[1,1],[1,-1]])/np.sqrt(2)

#Haar-random unitary
def haar_unitary(d):
    A, B = np.random.normal(size=(d, d)), np.random.normal(size=(d, d))
    Z = A + 1j * B

    Q, R = lg.qr(Z)

    Lambda = np.diag([R[i, i] / np.abs(R[i, i]) for i in range(d)])

    return Q @ Lambda @ Q

#corresponding function

U = haar_unitary(2**n)


def function_matrix(U,string):

  qml.QubitUnitary(U,list(range(num_qubits)))

  for i in range(num_qubits):
    obs[string[i]](i)

  qml.adjoint(qml.QubitUnitary(U,range(num_qubits)))


#############

n = 3
num_qubits = n

w = []
for i in range(2*n):
  w.append((np.random.rand()*2)-1)

def unitary(w):
  l = list(range(num_qubits))
  l.reverse()
  for i in l:
    qml.CNOT([i,(i+1)%num_qubits])
  for i in range(num_qubits):
    qml.RY(w[i],i)

def function(w,string):


  for i in range(num_qubits):
    qml.CNOT([i,(i+1)%num_qubits])
  for i in range(num_qubits):
    qml.RX(w[i],i)

  for i in range(num_qubits):
    obs[string[i]](i)



  for i in range(num_qubits):
    qml.adjoint(qml.RX(w[i],i))
  l = list(range(num_qubits))
  l.reverse()
  for i in l:
    qml.CNOT([i,(i+1)%num_qubits])




def prep_entangle(system):
    for wire in system:
        qml.Hadamard(wire)
        qml.CNOT(wires=[wire, wire + n])


#############################
# This code finds the Fourier coefficients with squared value larger than a threshold. It's an implementation of the Goldreich-Levin algorithm and can be expressed in terms of Quantum Statistical Queries.

dev = qml.device("default.qubit", wires=4*n+1, shots=None)

obs = [qml.Identity, qml.PauliX, qml.PauliY,qml.PauliZ]
string_p = [0,2,3]

@qml.qnode(dev)
def circuit_GL(string,k):

  prep_entangle(list(range(n)))
  #function(w,string_p)
  function_matrix(U,string_p)
  prep_entangle(list(range(2*n,3*n)))
  for i in range(n):
    obs[string[i]](2*n+i)
  qml.Hadamard(4*n)

  for i in range(k):
    qml.ctrl(qml.SWAP([i,i+2*n]),4*n)
    qml.ctrl(qml.SWAP([i+n,i+3*n]),4*n)

  qml.Hadamard(4*n)

  return qml.expval(qml.PauliZ(4*n))



dev = qml.device("default.qubit", wires=4*n+1, shots=None)
@qml.qnode(dev)
def circuit(string):

  prep_entangle(list(range(n)))
  #function(w,string_p)
  function_matrix(U,string_p)
  prep_entangle(list(range(2*n,3*n)))
  for i in range(n):
    obs[string[i]](2*n+i)
  qml.Hadamard(4*n)

  for i in range(2*n):
    qml.ctrl(qml.SWAP([i,i+2*n]),4*n,control_values=1)

  qml.Hadamard(4*n)

  return qml.expval(qml.PauliZ(4*n))

##############################
# It remains to estimate the sign. Given two Pauli strings, we can estimate if their signs are equal or not with a single QSQ. This allows us to learn all the coefficients up to a global sign.

dev_super = qml.device("default.qubit", wires=4*n+3)
@qml.qnode(dev_super)
def circuit_super(string1,string2,t):

  prep_entangle(list(range(n)))
  #function(w,string_p)
  function_matrix(U,string_p)
  prep_entangle(list(range(2*n,3*n)))
  qml.Hadamard(4*n+2)
  for i in range(n):
    qml.ctrl(obs[string1[i]](2*n+i),4*n+2,control_values=1)

  for i in range(n):
    qml.ctrl(obs[string2[i]](2*n+i),4*n+2,control_values=0)

  qml.Hadamard(4*n+2)

  if(t==1):
    qml.PauliX(4*n+1)


  qml.Hadamard(4*n)

  for i in range(2*n):
    qml.ctrl(qml.SWAP([i,i+2*n]),4*n,control_values=1)

  qml.ctrl(qml.SWAP([4*n+2,4*n+1]),4*n,control_values=1)
  qml.Hadamard(4*n)

  return qml.expval(qml.PauliZ(4*n))

obs = [qml.Identity, qml.PauliX, qml.PauliY,qml.PauliZ]



###############

from scipy.special import errstate

string = [3,3,3]
string_p = string
eps = [0.0001]
#err = []

for tau in eps:
  k = 0
  print(tau)
  for i in range(10):
    print(i)

    U = haar_unitary(2**n)

    L4 = []
    L4 = GL_alg(tau)
    coeff_L = abs_value(L4)
    signed = signed_coeff(coeff_L,L4)
    A = unitary_reconstruction(1,L4,signed)

    rho = np.zeros((2**n,2**n))
    rho[0][0] = 1
    x = np.trace(np.matmul(qml.matrix(A),np.array(rho)))

    P = qml.matrix(obs[string[0]](0)@obs[string[1]](1)@obs[string[2]](2))

    y = np.trace(np.matmul(np.matmul(U,rho),np.matmul(U.conjugate().transpose(),P)))

    k = k + np.abs(np.abs(x)-np.abs(y))
  err.append(k/10)
