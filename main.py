from predict import *
from states import *
from obs import *

#Unitary to be predicted
n = 6

U_id = np.eye(2**n)

U_had = [1]
for i in range(n):
    U_had= np.kron(U_had,Hadamard)

U_haar = haar_unitary(2**n)

unitaries = [U_id,U_had,U_haar]

# O = {'111111':0.25,'222222':0.25,'333333':0.25,'000000':0.25}
O = classical_obs(n);
print(O)

#Target state distribution
size_D = 100

#Uniformly-random product states
# D = haar_states(size_D,n)
# print('D is a set of',size_D,' uniformly random',n,'-qubit Pauli product states')

D_classical = classical_states(n)
D_pauli = pauli_product_states(size_D,n)
D_haar = haar_states(size_D,n)

D = [D_classical,D_pauli,D_haar]
distr = ['classical','pauli','haar']

step = 200
N_min = 200
N_max = 5000
eps = 0.9

num_shots = 10

for U in unitaries:
    print(U)
    for d in range(len(D)):
        for N in range(N_min,N_max+1,step):
            error = 0.0
            for i in range(num_shots):
                x = learn(U,O,N,n,eps)
                for rho in D[d]:
                    pred_val = pred(x,rho)
                    # print('Predicted value:',pred_val)

                    true_val = true_value(O,U,rho)
                    # print('True value:',true_val)
                    
                    error += (true_val-pred_val)**2
            error /= (num_shots*len(D[d]))
            print("Distribution = ", distr[d],"N = ",N,", error = ",error)