import pennylane as qml
from pennylane import numpy as np

# Prepare entangled state on system and copy
n = 4
def prep_entangle(system):
    for wire in system:
        qml.Hadamard(wire)
        qml.CNOT(wires=[wire, wire + n])

# generate a random vector for the weights:

w = []
for i in range(8):
  w.append((np.random.rand()*2)-1)

# generate a random vector for the pauli components

obs = [qml.Identity, qml.PauliX, qml.PauliY,qml.PauliZ]

pauli = []
for i in range(8):
  pauli.append(np.random.randint(4))

# define the gates
gate = []

for i in range(4):
  x = qml.s_prod(1/np.sqrt(w[2*i]**2+w[2*i+1]**2),qml.sum(qml.s_prod(w[2*i],obs[pauli[2*i]](i)),qml.s_prod(w[2*i+1],obs[pauli[2*i+1]](i))))
  gate.append(x)


def choi_state(system):
    prep_entangle(system)

    for i in range(4):
      qml.QubitUnitary(qml.matrix(gate[i]), i)


# This is the target Quantum Boolean Function
H = gate[0]@gate[1]@gate[2]@gate[3]

# sanity check
qml.is_hermitian(H)

#############################
# This code finds the Fourier coefficients with squared value larger than a threshold. It's an implementation of the Goldreich-Levin algorithm and can be expressed in terms of Quantum Statistical Queries.

dev = qml.device("default.qubit", wires=4*n+1)

obs = [qml.Identity, qml.PauliX, qml.PauliY,qml.PauliZ]

@qml.qnode(dev)
def circuit_GL(string,k):

  choi_state(list(range(0,n)))
  prep_entangle(list(range(2*n,3*n)))
  obs[string[0]](2*n)
  obs[string[1]](2*n+1)
  obs[string[2]](2*n+2)
  obs[string[3]](2*n+3)
  qml.Hadamard(4*n)

  for i in range(k):
    qml.ctrl(qml.SWAP([i,i+2*n]),4*n)
    qml.ctrl(qml.SWAP([i+n,i+3*n]),4*n)

  qml.Hadamard(4*n)

  return qml.expval(qml.PauliZ(4*n))

L1 = []
L2 = []
L3 = []
L4 = []


for j in range(4):
    t = circuit_GL([j,0,0,0],1)
    if (t>0.001):
      L1.append(j)
for j in L1:
    for l in range(4):
      t = circuit_GL([j,l,0,0],2)
      if (t>0.001):
        L2.append([j,l])
for j in L2:
   for l in range(4):
    t = circuit_GL([j[0],j[1],l,0],3)
    if (t>0.001):
        L3.append([j[0],j[1],l])
for j in L3:
   for l in range(4):
    t = circuit_GL([j[0],j[1],j[2],l],4)
    if (t>0.001):
        L4.append([j[0],j[1],j[2],l])

# Now that we know the Pauli strings with higher "weight", we can estimate the absolute value of their coefficient with a single QStat query

dev = qml.device("default.qubit", wires=4*n+1)
@qml.qnode(dev)
def circuit(string):

  choi_state(list(range(0,n)))
  prep_entangle(list(range(2*n,3*n)))
  obs[string[0]](2*n)
  obs[string[1]](2*n+1)
  obs[string[2]](2*n+2)
  obs[string[3]](2*n+3)
  qml.Hadamard(4*n)

  for i in range(2*n):
    qml.ctrl(qml.SWAP([i,i+2*n]),4*n,control_values=1)

  qml.Hadamard(4*n)

  return qml.expval(qml.PauliZ(4*n))

coeff_L = []

for i in L4:
  coeff_L.append(np.sqrt(circuit(i)))

# It remains to estimate the sign. Given two Pauli strings, we can estimate if their signs are equal or not with a single QSQ. This allows us to learn all the coefficients up to a global sign.

dev_super = qml.device("default.qubit", wires=4*n+3)
@qml.qnode(dev_super)
def circuit_super(string1,string2,t):

  choi_state(list(range(0,n)))
  prep_entangle(list(range(2*n,3*n)))
  qml.Hadamard(4*n+2)
  qml.ctrl(obs[string1[0]](2*n),4*n+2,control_values=1)
  qml.ctrl(obs[string1[1]](2*n+1),4*n+2,control_values = 1)
  qml.ctrl(obs[string1[2]](2*n+2),4*n+2,control_values =1)
  qml.ctrl(obs[string1[3]](2*n+3),4*n+2,control_values = 1)

  qml.ctrl(obs[string2[0]](2*n),4*n+2,control_values=0)
  qml.ctrl(obs[string2[1]](2*n+1),4*n+2,control_values = 0)
  qml.ctrl(obs[string2[2]](2*n+2),4*n+2,control_values =0)
  qml.ctrl(obs[string2[3]](2*n+3),4*n+2,control_values = 0)

  qml.Hadamard(4*n+2)
  
  if(t==1):
    qml.PauliX(4*n+1)


  qml.Hadamard(4*n)

  for i in range(2*n):
    qml.ctrl(qml.SWAP([i,i+2*n]),4*n,control_values=1)
  qml.ctrl(qml.SWAP([4*n+2,4*n+1]),4*n,control_values=1)

  qml.Hadamard(4*n)

  return qml.expval(qml.PauliZ(4*n))

signed_coeff = []

for i in range(len(L4)):
  print("i =",i)
  x = circuit_super(L4[0],L4[i],0)
  y = circuit_super(L4[0],L4[i],1)
  if(x<y):
    signed_coeff.append(-coeff_L[i])
  else:
    signed_coeff.append(coeff_L[i])

#Finally, we estimate a unitary U such that either U or -U is close to H in operator 2-norm

def operator_2_norm(R):

    return np.sqrt(np.trace(R.conjugate().transpose() @ R))

def unitary_reconstruction(t):
  U = qml.s_prod(scalar=np.array(0), operator = obs[0](0)@obs[0](1)@obs[0](2)@obs[0](3))
  for i in range(len(L4)):
        U = qml.sum(U,qml.s_prod(t*signed_coeff[i],obs[L4[i][0]](0)@obs[L4[i][1]](1)@obs[L4[i][2]](2)@obs[L4[i][3]](3)))
  return U

U1 = unitary_reconstruction(1)
U2 = unitary_reconstruction(-1)

dist1 = operator_2_norm(qml.matrix(H)- qml.matrix(U1))
dist2 = operator_2_norm(qml.matrix(H)- qml.matrix(U2))

#so either dist1 or dist2 will be small



