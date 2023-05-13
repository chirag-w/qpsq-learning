#Define functions to generate various sets of states

from util import *
import numpy.linalg as lg
import cmath


Hadamard = np.array([[1,1],[1,-1]])/np.sqrt(2)

#Haar-random unitary
def haar_unitary(d):
    A, B = np.random.normal(size=(d, d)), np.random.normal(size=(d, d))
    Z = A + 1j * B

    Q, R = lg.qr(Z)

    Lambda = np.diag([R[i, i] / np.abs(R[i, i]) for i in range(d)])

    return Q @ Lambda @ Q

#Unitary for Quantum Fourier Transform
def qft_unitary(N):
    omega = cmath.exp(2*cmath.pi*1j/N)
    U = np.zeros((N,N), dtype = complex)
    for i in range(N):
        for j in range(N):
            U[i,j] = omega**(i*j)
    U /= np.sqrt(N)
    return U
    

#Return a random permutation function and its unitary
def random_permutation_unitary(N):
    init = np.array(range(N))
    shuffled = np.random.permutation(init)
    unitary = np.zeros((N,N), dtype = float)
    for i in range(N):
        unitary[i][shuffled[i]] = 1
    return shuffled,unitary

