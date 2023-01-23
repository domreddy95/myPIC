import numpy as np
import matplotlib.pyplot as plt


N = 1000
X = np.linspace(0,1,num=N)
Y = np.zeros((N))
points = np.linspace(0,1,num=11)
pointsy = np.zeros_like(points)
partx = np.random.rand(N)
party = (np.zeros_like(partx))+0.0005


plt.plot(X,Y)
plt.scatter(points,pointsy)
plt.scatter(partx,party,s=8)
plt.title("Simulation Setup")
plt.show()