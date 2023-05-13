#Define functions to generate various sets of states

from util import *
from unitaries import *

#Uniformly-random product states
def pauli_product_states(N,n):
    D = []
    for i in range(N):
        rho = [1]
        indices = np.random.randint(0,6,n)
        for i in range(n):
            rho = np.kron(rho,pauli_states[indices[i]])
        D.append(rho)
    return D

#Maximally mixed state
def max_mixed_state(n):
    return np.eye(2**n)/2**n

#GHZ State
def ghz_state(n):
    rho = np.zeros((2**n,2**n), dtype = complex)
    rho[0][0] = rho[2**n-1][2**n-1] = rho[2**n-1][0] = rho[0][2**n-1] = 0.5
    return rho

#Haar-random states
def haar_states(N,n):
    D = []
    zero_state = np.zeros((2**n,2**n))
    zero_state[0,0] = 1
    for i in range(N):
        haar = haar_unitary(2**n)
        rho = haar @ zero_state @ haar.conjugate().transpose()
        D.append(rho)
    return D

#All n-qubit computational basis states
def classical_states(n):
    D = []
    for i in range(2**n):
        rho = np.zeros((2**n,2**n),dtype = complex)
        rho[i,i] = 1
        D.append(rho)
    return D

#LWE Sample states -> 1/(root(q^n)) \Sum (|a>|a.s+e_a>)
def lwe_samples(num_states, s , n , q, error_distr):
    # (n+1)-qudit states
    # a.s + e_a
    states = []
    for i in range(num_states):
        statevector = np.zeros(q**(n+1), dtype = complex)
        for input in range(q**n):
            inp = input
            a = np.zeros(n, dtype = int)
            for index in range(n):
                a[-index] = inp%q
                inp = int(inp%q)
            dot = 0
            for index in range(n):
                dot += int(s[index])*a[index]
            out = (dot + np.random.choice(range(q),p = error_distr))%q
            state = input*q+out
            statevector[state] = 1
        statevector/= np.sqrt(q**n)
        states.append(statevector)
    return density_matrix(states)