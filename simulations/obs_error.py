#Plot errors in two methods of constructing QSQ oracles
#1 - Classical shadows for measuring observables
#2 - Norally distributed error

from coeff import *
import seaborn as sns
import matplotlib.pyplot as plt

def get_mult_samples_input_states(n, num_samples, N):
    init_states = np.random.randint(0,6,(N,n))
    input_states = []
    for i in range(N):
        for j in range(num_samples):
            input_states.append(init_states[i])
    return input_states

def classical_shadow_error(n, num_samples, U, O, N):
    input_states = get_mult_samples_input_states(n,num_samples,N)
    output_states = apply_unitary_measure(UnitaryGate(U), input_states)
    
    Udg = U.conjugate().transpose()

    exp_est_all = shadow_observable(O,output_states)
    exp_est = []
    for i in range(N):
        exp_est.append(np.mean(exp_est_all[i*num_samples:(i+1)*num_samples]))
    
    dev = []
    for i in range(N):
        rho = [1]
        for j in range(n):
            rho = np.kron(rho,pauli_states[input_states[i*num_samples][j]])
        output_i = U @ rho @ Udg
        # output_states.append(output_i)
        
        exp_i = 0
        for P in O:
            coeff = O[P] #Coefficient of Pauli P in pauli basis representation of O
            obs = 1
            for ind in range(len(P)):
                obs = np.kron(obs,pauli_obs[int(P[ind])])
            exp_i+= coeff*np.trace(obs@output_i).real
        dev.append(exp_i-exp_est[i])
    return dev

n = 1
delta = 0.0455
tau = 0.2

O = pauli_one_obs(n,0)
U = haar_unitary(2**n)

N = 1000

local = 1 #Maximum number of non-Identity qubits in any term in Pauli expansion of O
coeff = 1 #Sum of squares of coefficients of Pauli expansion

num_samples = int(np.ceil((3**local) * coeff / (delta * tau * tau)))
root_samples = np.sqrt(num_samples)


D1 = classical_shadow_error(n, num_samples, U, O, N)
D2 = np.random.normal(0, tau/2*root_samples, N)/root_samples



sns.histplot(D1, label = 'Classical Shadow',bins = 35, binrange = (-0.35,0.35), color = 'r')
sns.histplot(D2, label = 'Normal Distr', bins = 35, binrange = (-0.35,0.35), color = 'b')

plt.xlabel('error')
plt.legend(loc = 'best')
plt.title('Comparing deviations using two methods')
plt.show()

sns.histplot(D1, label = 'Classical Shadow', color = 'r')
sns.histplot(D2, label = 'Normal Distr',  color = 'b')

plt.xlabel('error')
plt.legend(loc = 'best')
plt.title('Comparing deviations using two methods')
plt.show()