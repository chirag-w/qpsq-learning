import sys

sys.path.append('../')

from predict import *
from states import *
from speck import *
from obs import *

from sklearn.metrics import confusion_matrix

n = 6
eps = 0.9

str_form = '0'+str(n)+'b'

#Function to forge a classical protocol
def forge(U,f, N,n, target_messages = []):

    #Construct observables 
    O = classical_obs(n)

    #Learn to predict output of the classical protocol
    model = learn(U,O,N,n,eps)
    #print(model)
    target_states = []
    for message in target_messages:
        #Message is a binary string
        #Convert to density matrices for prediction
        rho = [1]
        for i in range(n):
            rho = np.kron(rho,pauli_states[int(message[i])])
        target_states.append(rho)

    forged_output = []
    error = 0.0
    for index in range(len(target_states)):
        bits = ''
        expected_bits = format(f[int(target_messages[index],2)], str_form)
        #print(f[int(target_messages[index],2)])
        pred_val = (2**n)*pred(model,target_states[index])
        if pred_val < 0:
            pred_val = 0
        if pred_val >= 2**n:
            pred_val = 2**n-1
        #print(pred_val)
        pred_bits = format(round(pred_val), str_form)
        
        for i in range(n):
            if pred_bits[i]!= expected_bits[i]:
                error += 1
        forged_output.append(bits)
        # print('Input:',target_messages[index])
        # print('Expected output:', expected_bits)
        # print('Predicted Output:',bits)
    error /= n 
    error /= 2**n
    #print('Prediction error:', error)
    return forged_output,error

messages = []

for i in range(2**n):
    messages.append(format(i, str_form))

samples = [5000,10000,15000,20000,25000]

#Random permutation
f,U = random_permutation_unitary(2**n)
print('f is a random permutation')
for N in samples:
    forged_messages,err = forge(U,f,N,n, messages)
    print(N,'samples, error =',err)

#Speck
f,U = speck_unitary(1, (3,3,3,3))
print('f is the reduced version of single-round speck')
for N in samples:
    forged_messages,err = forge(U,f,N,n, messages)
    print(N,'samples, error =',err)

#Identity
f,U = (range(2**n), np.eye(2**n))
print('f is the identity function') 
for N in samples:
    forged_messages,err = forge(U,f,N,n, messages)
    print(N,'samples, error =',err)
