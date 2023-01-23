import numpy as np
import matplotlib.pyplot as plt


N = 200000
vt = 1


np.random.seed(3453)
Vx = np.random.normal(loc=0,scale=1,size=N)*vt
np.random.seed(4554)
Vy = np.random.normal(loc=0,scale=1,size=N)*vt
V = ((Vx**2)+(Vy**2))**0.5
plt.figure(1)
plt.hist(V,bins=1000)
plt.title("Maxwell Velocity Disribution, Vt = 1")
plt.xlabel("V")
plt.ylabel("N")
plt.figure(2)
plt.hist(Vx,bins=1000)
plt.title("Velocity Components in Z direction, Vt = 1")
plt.xlabel("Vz")
plt.ylabel("N")
plt.figure(3)
plt.hist(Vy,bins=1000)
plt.title("Velocity Components in R direction, Vt = 1")
plt.xlabel("Vr")
plt.ylabel("N")
plt.figure(4)
#plt.scatter(Vx,Vy,s=1)
plt.show()