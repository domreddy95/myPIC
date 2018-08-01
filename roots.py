import numpy as np
import matplotlib.pyplot as plt

L = 100.0
vb = 1.0
wp = 1.0
N = 15
mode = np.linspace(1,N,N)
#print(mode)
k = (2*np.pi*mode)/L
growth_rate = np.zeros_like(mode) 
for i in range (0,N):
    a0 = 1
    a1 = -2*k[i]*vb
    a2 = (((k[i]*vb)**2)-2*wp**2)
    a3 = 2*(wp**2)*k[i]*vb
    a4 = -(wp*k[i]*vb)**2
    dispersion_relation = [a0,a1,a2,a3,a4]
    roots = np.roots(dispersion_relation)
    #print(np.imag(roots[3]))
    growth_rate[i] = np.imag(roots[2])
    #print(growth_rate[i])
print(growth_rate)
t = np.linspace(0,10,20)
T = len(t)
coeff = np.zeros((N,T))
for i in range (0,T):
    for k in range (0,N):
        coeff[k,i] = np.exp(growth_rate[k]*t[i])

print()
#print(coeff)
x = np.linspace(0,L,1001)
z = np.sin(x)
print(np.sum(z**2))
energy = np.zeros((T))
for i in range (0,T):
    for j in range (0,N):
        z = np.sin((2.0*np.pi*mode[j]*x)/L)
        y = (coeff[j,i])*z
        energy[i] += (np.sum(y**2)/1001)

#print(energy)
#plt.figure()
#plt.plot(t,energy)
#plt.show()