import sys

sys.path.append('../')

from predict import *
from speck import *
from obs import *

from sklearn.metrics import confusion_matrix

n = 6
eps = 0.9

# f,U = random_permutation_unitary(2**n)
# print(f)

f,U = speck_unitary(1, (3,3,3,3))

D = []

O = classical_obs(n)

str_form = '0'+str(n)+'b'

N = 10000
eps = 0.9

model = learn(U,O,N,n,eps)

num_shots = 100

b = np.random.randint(2, size = num_shots)
x = np.random.randint(2**n, size = num_shots)
b_pred = []
threshold = 0.055
correct = 0
for i in range(num_shots):
    output = -1
    y = 0
    if b[i] == 1:
        output = np.random.randint(2**n)
    else:
        output = f[x[i]]

    message = format(output,str_form)
    rho = [1]
    for j in range(n):
        rho = np.kron(rho,pauli_states[int(message[j])])

    I = np.eye(2**n)
    y = true_value(O,I,rho)
    
    message = format(x[i],str_form)
    rho = [1]
    for j in range(n):
        rho = np.kron(rho,pauli_states[int(message[j])])
    y_pred = pred(model,rho)
    print(y,y_pred)

    dist = (y-y_pred)**2
    if dist > threshold:
        b_pred.append(1)
    else:
        b_pred.append(0)
    if b_pred[i] == b[i]:
        correct+=1

print('Accuracy =', correct/num_shots)
print(confusion_matrix(b,b_pred)/num_shots)
