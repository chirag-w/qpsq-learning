import numpy as np 
import matplotlib.pyplot as plt

d_str = ['comp','stab','haar']
o_str = 'z1'

n = 6
eps = 0.9
tolerance = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25])
step = 50
N_min = 50
N_max = 1000
samples = range(N_min,N_max+1,step)
n_reps = 10

suffix = "_eps"+str(eps)+"_reps"+str(n_reps)+"_"+o_str+"_n"+str(n)+"_N"+str(N_min)+"-"+str(N_max)+".npy"
data0 = np.load("haar-unitaries"+"_"+d_str[0]+suffix)
data1 = np.load("haar-unitaries"+"_"+d_str[1]+suffix)
data2 = np.load("haar-unitaries"+"_"+d_str[2]+suffix)

# plt.plot(samples,data0[1], label = 'Target states: '+d_str[0])
# plt.plot(samples,data1[1], label = 'Target states: '+d_str[1])
# plt.plot(samples,data2[1], label = 'Target states: '+d_str[2])

# plt.xlabel('Number of samples')
# plt.ylabel('SSE')
# plt.title('SQ Learning Error for Haar-random Unitary, tau = '+str(tolerance[1])+',eps ='+str(eps)+',n ='+str(n))
# plt.legend(loc = 'upper right')
# plt.show()

for i in [1,3,5]:
    plt.plot(samples[0:10],data0[i][0:10], label = 'Tolerance: '+str(tolerance[i]))

plt.xlabel('Number of queries')
plt.ylabel('Error')
plt.title('Learning error for haar-random unitaries, Distr = '+d_str[0]+',n ='+str(n))
plt.legend(loc = 'upper right')
plt.show()

for i in [1,3,5]:
    plt.plot(samples[0:10],data1[i][0:10], label = 'Tolerance: '+str(tolerance[i]))

plt.xlabel('Number of queries')
plt.ylabel('Error')
plt.title('Learning error for haar-random unitaries, Distr = '+d_str[1]+',n ='+str(n))
plt.legend(loc = 'upper right')
plt.show()

for i in [1,3,5]:
    plt.plot(samples[0:10],data2[i][0:10], label = 'Tolerance: '+str(tolerance[i]))

plt.xlabel('Number of queries')
plt.ylabel('Error')
plt.title('Learning error for haar-random unitaries, Distr = '+d_str[2]+',n ='+str(n))
plt.legend(loc = 'upper right')
plt.show()