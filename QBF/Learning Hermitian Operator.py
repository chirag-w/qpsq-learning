!pip install pennylane
import pennylane as qml
from pennylane import numpy as np

##################################

# Prepare entangled state on system and copy
n = 3
def prep_entangle(system):
    for wire in system:
        qml.Hadamard(wire)
        qml.CNOT(wires=[wire, wire + n])

obs = [qml.Identity, qml.PauliX, qml.PauliY,qml.PauliZ]

def choi_state(system,gate):
    prep_entangle(system)

    for i in range(n):
      qml.QubitUnitary(qml.matrix(gate[i]), i)


def gate_gen():

  # generate a random vector for the weights:
  w = []
  for i in range(2*n):
    w.append((np.random.rand()*2)-1)

  # generate a random vector for the pauli components
  pauli = []
  for i in range(n):
    ran = np.random.randint(4)
    pauli.append(ran)
    pauli.append((ran + 1 + np.random.randint(3)) % 4)

  # define the gates
  gate = []

  for i in range(n):
    x = qml.s_prod(1/np.sqrt(w[2*i]**2+w[2*i+1]**2),qml.sum(qml.s_prod(w[2*i],obs[pauli[2*i]](i)),qml.s_prod(w[2*i+1],obs[pauli[2*i+1]](i))))
    gate.append(x)

  return gate

#####################

# This code finds the Fourier coefficients with squared value larger than a threshold. It's an implementation of the Goldreich-Levin algorithm and can be expressed in terms of Quantum Statistical Queries.

dev = qml.device("default.qubit", wires=4*n+1, shots=None)

obs = [qml.Identity, qml.PauliX, qml.PauliY,qml.PauliZ]

@qml.qnode(dev)
def circuit_GL(string,k):

  choi_state(list(range(0,n)),gate)
  prep_entangle(list(range(2*n,3*n)))
  for i in range(n):
    obs[string[i]](2*n+i)
  qml.Hadamard(4*n)

  for i in range(k):
    qml.ctrl(qml.SWAP([i,i+2*n]),4*n)
    qml.ctrl(qml.SWAP([i+n,i+3*n]),4*n)

  qml.Hadamard(4*n)

  return qml.expval(qml.PauliZ(4*n))

def GL_alg(eps):

  L = []
  for i in range(n):
    L.append([])


  for k in range(4):
    string = []
    string.append(k)
    string.extend([0]*(n-1))
    dev = np.random.normal(0, eps*0.1, 1)[0] 
    t = circuit_GL(string,1) +dev
    if (t>eps):
      L[0].append([k])

  for i in range(n-1):
    for j in L[i]:
      for k in range(4):
        string = []
        string.extend(j)
        string.append(k)
        string.extend([0]*(n-i-2))
        dev = np.random.normal(0, eps*0.1, 1)[0] 
        t = circuit_GL(string,i+2) +dev
        if (t>eps):
          string = []
          string.extend(j)
          string.append(k)
          L[i+1].append(string)

  return L[n-1]


##########################


# Now that we know the Pauli strings with higher "weight", we can estimate the absolute value of their coefficient with a single QStat query

dev = qml.device("default.qubit", wires=4*n+1, shots=None)
@qml.qnode(dev)
def circuit(string):

  choi_state(list(range(0,n)),gate)
  prep_entangle(list(range(2*n,3*n)))
  for i in range(n):
    obs[string[i]](2*n+i)
  qml.Hadamard(4*n)

  for i in range(2*n):
    qml.ctrl(qml.SWAP([i,i+2*n]),4*n,control_values=1)

  qml.Hadamard(4*n)

  return qml.expval(qml.PauliZ(4*n))

def abs_value(L4,eps):
  coeff_L = []

  for i in L4:
    dev = np.random.normal(0, eps*0.1, 1)[0] 
    t = circuit(i) +dev
    coeff_L.append(np.sqrt(t))


  return coeff_L

##############################
# It remains to estimate the sign. Given two Pauli strings, we can estimate if their signs are equal or not with a single QSQ. This allows us to learn all the coefficients up to a global sign.

dev_super = qml.device("default.qubit", wires=4*n+3)
@qml.qnode(dev_super)
def circuit_super(string1,string2,t):

  choi_state(list(range(0,n)),gate)
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

def signed_coeff(coeff_L,L4,eps):
  signed_coeff = []
  k = np.argmax(coeff_L)
  for i in range(len(L4)):
    #print("i =",i)
    dev = np.random.normal(0, eps*0.1, 1)[0] 
    x = circuit_super(L4[k],L4[i],0) +dev
    dev = np.random.normal(0, eps*0.1, 1)[0] 
    y = circuit_super(L4[k],L4[i],1) +dev
    if(x<y):
      signed_coeff.append(-coeff_L[i])
    else:
      signed_coeff.append(coeff_L[i])

  return signed_coeff

#Finally, we estimate a unitary U such that either U or -U is close to H in operator 2-norm

def operator_2_norm(R):
    x= qml.math.dot(R.conjugate().transpose(), R)
    return np.sqrt(np.trace(x))

def unitary_reconstruction(t,L4,signed):
  U = obs[0](0)
  for i in range(1,n):
    U = U@obs[0](i)
  U = qml.s_prod(scalar=np.array(0), operator = U)
  for i in range(len(L4)):
        op = obs[L4[i][0]](0)
        for j in range(1,n):
            op = op@obs[L4[i][j]](j)
        U = qml.sum(U,qml.s_prod(t*signed[i],op))
  return U


def check(L4,signed,H):

  U1 = unitary_reconstruction(1,L4,signed)
  U2 = unitary_reconstruction(-1,L4,signed)

  dist1 = operator_2_norm(qml.matrix(H)- qml.matrix(U1))
  dist2 = operator_2_norm(qml.matrix(H)- qml.matrix(U2))

  #so either dist1 or dist2 will be small
  print("positive: ",round(np.real(dist1),4))
  print("negative: ",round(np.real(dist2),4))
  print()

  return float(min(np.real(dist2),np.real(dist1)))


#Now we check the code for 15 random hermitian unitaries:
eps =0.01
k = 0
for i in range(15):
  print(i)
  gate = gate_gen()
  # This is the target Quantum Boolean Function
  H = gate[0]@gate[1]@gate[2]
  L4 = []
  L4 = GL_alg(eps)
  coeff_L = abs_value(L4,eps)
  signed = signed_coeff(coeff_L,L4,eps)
  k = k + check(L4,signed,H)
  print("k = ", k)
  print()
