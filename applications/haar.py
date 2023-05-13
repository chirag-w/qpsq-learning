import sys

sys.path.append('../')

from predict import *
from states import *
from obs import *

#Unitary to be predicted
n = 6

U = haar_unitary(2**n)

O = pauli_one_obs(n,0,'Z') # Z on qubit 0

# O = classical_obs(n)

#Target state distribution
size_D = 100

#n-qubit Haar-random states
D = haar_states(size_D,n)

#Uniformly-random product states
# D = pauli_product_states(size_D,n)

#Classical states
# D = classical_states(n)
# size_D = len(D)

step = 500
N_min = 500
N_max = 10000
eps = 0.9

for N in range(N_min,N_max+1,step):
    x = learn(U,O,N,n,eps)
    # print(x)
    error = 0.0
    for rho in D:
        pred_val = pred(x,rho)
        # print('Predicted value:',pred_val)

        true_val = true_value(O,U,rho)
        # print('True value:',true_val)
        
        error += (true_val-pred_val)**2
    error /= size_D
    print("N = ",N,", error = ",error)
