import sys

sys.path.append('../')

from predict import *
from states import *
from obs import *

n = 6

str_form = '0'+str(n)+'b'

def dot(x,s):
    count = 0
    for i in range(n):
        if x[i] == '1' and s[i] == '1':
            count+=1
    count = count % 2
    return count

s = '111111'

U_f = np.eye(2**n, dtype = complex)
for i in range(2**n):
    if dot(format(i,str_form),s) == 1:
        U_f[i,i] *= -1


error = [2/3,1/3]

for i in range(2**n):    
    flip = np.random.choice([0,1],p = error)
    if flip == 1:
        U_f[i,i] *= -1


H_t = [1]
for i in range(n):
    H_t = np.kron(H_t, Hadamard)
U = H_t @ U_f 

target_state = np.ones((2**n,2**n))/2**n

step = 500
N_min = 500
N_max = 10000
eps = 0.9

O = classical_obs(n)

for N in range(N_min,N_max+1,step):
    x = learn(U,O,N,n,eps)

    pred_val = 2**n * pred(x,target_state)

    true_val = 2**n * true_value(O,U,target_state)

    print(N,'samples :',pred_val,true_val)