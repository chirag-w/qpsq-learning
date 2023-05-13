#Generate various observables
from util import *

#Identity
def id_obs(n):
    obs_str = ''
    for i in range(n):
        obs_str += '3'
    O = {obs_str:1}
    return O

#Pauli observable on single qubit
def pauli_one_obs(n,pos, obs = 'Z'):
    obs_str = ''
    for i in range(pos):
        obs_str += '3'
    
    if obs == 'Z':
        obs_str += '0'
    elif obs == 'X':
        obs_str += '1'
    elif obs == 'Y':
        obs_str += '2'
    else:
        obs+= '3'

    for i in range(n-pos-1):
        obs_str += '3'
    O = {obs_str:1}
    return O 

#Pauli observable on all qubits
def pauli_all_obs(n, obs = 'Z'):
    c = ''
    if obs == 'Z':
        c = '0'
    elif obs == 'X':
        c = '1'
    elif obs == 'Y':
        c = '2'
    else:
        c = '3'
    obs_str = ''
    for i in range(n):
        obs_str += c
    O = {obs_str:1}
    return O

def classical_obs(n):
    #For a classical string s, <s|O|s> = s, upto a normalization factor
    # <s|O|s> = s/(2**n)
    id = ''
    for i in range(n):
        id+= '3'
    O = {id:(2**n-1)/(2**(n+1))}

    coeff = -0.5
    
    string = np.zeros(n,dtype = int)
    for i in range(n):
        string[i] = 3 # string represents identity

    for i in range(n):
        coeff /= 2
        string[i] = 0 #Use string to represent Z_i
        obs = ''.join(map(str,string))
        O[obs] = coeff
        string[i] = 3 #Restore string to identity
    
    return O