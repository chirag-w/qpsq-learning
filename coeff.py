from dataset import *
from itertools import *
import math

#Pauli indices
# 0 -> Z
# 1 -> X
# 2 -> Y
# 3 -> I
def generate_paulis(inp,n,k):
    #Generate all Pauli product observables with <= k non-identity terms
    #and tr(P@inp[i])!=0 for some i
    #stored as strings in {0,1,2,3}^n 
    s = ''.join(['3']*n)
    paulis = {s}
    for size in range(k+1):
        for ind in range(len(inp)):
            for posns in combinations(range(n),size):
                s = ['3']*n
                for pos in posns:
                    s[pos] = str(int(inp[ind][pos]/2))
                paulis.add(''.join(s))
    return paulis

def coeff_pauli(P,inp,k,obs_val,eta,eps_tilde = 0.0001):
    N = len(inp)
    n = len(inp[0])
    coeff = 0
    for l in range(N):
        zero = False
        sign = 1
        for i in range(n):
            if P[i] != '3' and str(int(inp[l][i]/2)) != P[i]:
                #Trace is zero 
                zero = True
                break
            elif P[i] != '3' and inp[l][i]%2 == 1: 
                #This qubit is in negative eigenstate
                sign *= -1
            else:
                #This state is either in positive eigenstate or the Pauli is identity   
                continue
        if zero:
            continue
        else:
            coeff += sign*obs_val[l]
    coeff = coeff/N
    mod = mod_pauli(P)
    
    if (1/3)**mod < 2*eps_tilde or abs(coeff) < (2 * (3**(mod/2)) * np.sqrt(eps_tilde) * eta):
        return 0
    return coeff * (3**mod)

def sum_coeff(O):
    #Sum of absolute values of the coefficients of O in the Pauli basis
    sum = 0
    for obs in O:
        sum+= abs(O[obs])
    return sum

def mod_pauli(P):
    #Number of non-identity terms in a Pauli product observable
    count = 0
    for obs in P:
        if obs!='3':
            count+=1
    return count

def shadow_observable(O,shadow):
    #Placeholder function
    N = len(shadow)
    n = len(shadow[0])
    val = np.zeros(N, dtype = float)
    for l in range(N):
        for obs in O:
            prod = O[obs]
            for i in range(n):
                if obs[i] == '3': #If observable is identity, trace is 1
                    continue
                elif str(int(shadow[l][i]/2)) != obs[i]: #If observable doesn't match, trace is 0
                    prod*= 0
                elif shadow[l][i]%2 == 1: #If observable matches but the eigenvalue is negative,  trace is -3
                    prod*= -3
                else: #Observable matches and positive eigenstate, trace is 3
                    prod*= 3
            val[l] += prod
    return val

def degree(O,n):
    #Degree of the n-qubit observable
    count = np.zeros(n)
    for obs in O:
        for qubit in range(n):
            if obs[qubit]!='3':
                count[qubit]+=1
    return int(max(count))

def C_k_d(k,d):
    if d == 0:
        return 0
    numerator = np.sqrt(2 * math.factorial(k))
    denominator = np.sqrt(d) * (k**(k+2.5)) * ((2 * np.sqrt(6) + 4 * np.sqrt(3))**k)
    return numerator/denominator

def hyperparams(eps,O,n):
    k = int(np.ceil(np.log(2/eps)/np.log(1.5)))
    C = C_k_d(k,degree(O,n))
    num = eps * C * C
    denom = 81 * (2**(k+1)) * (n**k)
    eps_tilde = num/denom
    return k, eps_tilde

def learn(U,O,N,n,eps, construct_data = True, inp = None, out = None, flag = 0, tau = 0):
    k,eps_tilde = hyperparams(eps,O,n)
    # print(k,eps_tilde)
    # eps_tilde = 0

    if flag == 0:
        #Classical Shadow Tomography (with random Pauli measurements)
        if construct_data:  #If dataset not provided already
            inp, out = construct_dataset(U,N,n) #Construct dataset consisting of single-qubit stabilizer states
        obs_val = shadow_observable(O,out) #Compute output of observable on shadow outputs 

    else:
        #Quantum Statistical Query
        if construct_data:
            inp, out = construct_dataset_qsq(U,O,N,n,tau)  
        obs_val = out

    paulis = generate_paulis(inp,n,k)
    x = {} #Coefficients of the approximate k-truncated observable
    eta = sum_coeff(O)
    for P in paulis:
        coeff = coeff_pauli(P,inp,k,obs_val,eta, eps_tilde)
        if coeff!= 0 :
            x[P] = coeff
    return x 