#Learning haar-random unitaries for various tolerances

import sys

sys.path.append('../')

from predict import *
from states import *
from obs import *

#Unitary to be predicted
n = 6
n_reps = 10

unitaries = []
for i in range(n_reps):
    unitaries.append(haar_unitary(2**n))

O = pauli_one_obs(n, 0)
o_str = 'z1'
print('O =', o_str)

#Target state distribution
size_D = 100

D_classical = classical_states(n)
D_pauli = pauli_product_states(size_D,n)
D_haar = haar_states(size_D,n)

D = [D_classical,D_pauli,D_haar]
d_str = ['comp','stab','haar']

eps = 0.9
tolerance = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25])
step = 50
N_min = 50
N_max = 1000
samples = range(N_min,N_max+1,step)

for d_iter in range(len(D)):
    data = np.zeros((len(tolerance),len(samples)))
    for t_iter in range(len(tolerance)):
        tau = tolerance[t_iter]
        #print('SQ, tau = ', tau)
        for N_iter in range(len(samples)):
            N = samples[N_iter]
            error = 0.0
            for u_iter in range(len(unitaries)):
                U = unitaries[u_iter]
                x = learn(U,O,N,n,eps, flag = 1, tau = tau)
                for rho in D[d_iter]:
                    pred_val = pred(x,rho)
                    # print('Predicted value:',pred_val)

                    true_val = true_value(O,U,rho)
                    # print('True value:',true_val)
                       
                    error += (true_val-pred_val)**2
            error /= (n_reps*len(D[d_iter]))
            print("Distribution = ", d_str[d_iter],", N = ",N,", tau = ",tau ,", n = ",n,", error = ",error)
            data[t_iter,N_iter] = error 
    filename = "haar-unitaries"+"_"+d_str[d_iter]+"_eps"+str(eps)+"_reps"+str(n_reps)+"_"+o_str+"_n"+str(n)+"_N"+str(N_min)+"-"+str(N_max)
    np.save(filename,data)



# print('Classical Shadow')
# for u in range(len(unitaries)):
#     U = unitaries[u]
#     for d in range(len(D)):
#         for N in range(N_min,N_max+1,step):
#             error = 0.0
#             for i in range(num_shots):
#                 x = learn(U,O,N,n,eps)
#                 for rho in D[d]:
#                     pred_val = pred(x,rho)
#                     # print('Predicted value:',pred_val)

#                     true_val = true_value(O,U,rho)
#                     # print('True value:',true_val)
                    
#                     error += (true_val-pred_val)**2
#             error /= (num_shots*len(D[d]))
#             print("Unitary = ", u_str[u],", Distribution = ", distr[d],", N = ",N,", n = ",n,", error = ",error)
