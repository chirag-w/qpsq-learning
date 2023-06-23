import numpy as np 
import matplotlib.pyplot as plt

u_str = ['identity','hadamard','haar_unitary']
d_str = ['classical','stabilizer','haar_states']
o_str = 'z1'


n = 6
eps = 0.9
tolerance = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
samples = np.array([100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000])
num_shots = 10

suffix = "_eps"+str(eps)+"_reps"+str(num_shots)+"_"+o_str+"_n"+str(n)+".npy"
data00 = np.load(u_str[0]+"_"+d_str[0]+suffix)
data01 = np.load(u_str[0]+"_"+d_str[1]+suffix)
data02 = np.load(u_str[0]+"_"+d_str[2]+suffix)

data10 = np.load(u_str[1]+"_"+d_str[0]+suffix)
data11 = np.load(u_str[1]+"_"+d_str[1]+suffix)
data12 = np.load(u_str[1]+"_"+d_str[2]+suffix)

data20 = np.load(u_str[2]+"_"+d_str[0]+suffix)
data21 = np.load(u_str[2]+"_"+d_str[1]+suffix)
data22 = np.load(u_str[2]+"_"+d_str[2]+suffix)

# plt.plot(samples,data02[1], label = 'Unitary: '+u_str[0])
# plt.plot(samples,data12[1], label = 'Unitary: '+u_str[1])
# plt.plot(samples,data22[1], label = 'Unitary: '+u_str[2])


# plt.xlabel('Number of samples')
# plt.ylabel('SSE')
# plt.title('SQ Learning Error with target haar-random states, tau = '+str(tolerance[1])+',eps ='+str(eps)+',n ='+str(n))
# plt.legend(loc = 'upper right')
# plt.show()

for i in range(len(tolerance)):
    plt.plot(samples,data20[i], label = 'Tolerance: '+str(tolerance[i]))

plt.xlabel('Samples')
plt.ylabel('SSE')
plt.title('SQ Learning Error for Haar Unitary, Distr = '+d_str[0]+',eps ='+str(eps)+',n ='+str(n))
plt.legend(loc = 'upper right')
plt.show()