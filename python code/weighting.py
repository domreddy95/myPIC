import numpy as np
import matplotlib.pyplot as plt
from numba import jit

charge = 1

N = 2

z = 2
r = 1
dz = 0.1
dr = 0.1
nz = int(z/dz)
nr = int(r/dr)

grid = np.zeros((nz+1,nr+1))
nodal_volume = np.ones((nz+1,nr+1))
np.random.seed(4744)
X = z*np.random.rand(N)
np.random.seed(3125)
Y = r*np.random.rand(N)
Pz = [0.5,1.5]
Pr = [0.5,0.5]
xt = 0.05
yt = 0.05
print(X,Y)

def nodalvolume(n):
    for i in range(0,nz+1):
        for j in range(0,nr+1):
            j_min = j-0.5
            j_max = j+0.5
            if (j_min<0): 
                j_min=0
            if (j_max>nr):
                j_max=nr
            a = 0.5 if (i==0 or i==nz) else 1
            n[i][j] = a*dz*(((dr*j_max)**2)-((dr*j_min)**2))

@jit(nopython=True)
def weighting(G,xx,yy,nv):
    for i in range (0,N):
        hx = xx[i]
        hy = yy[i]
        ix0 = int(hx/dz)
        ix1 = ix0 + 1
        iy0 = int(hy/dr)
        iy1 = iy0 + 1
        G[ix0,iy0] += (1-hx)*(1-hy)/nv[ix0][iy0]
        G[ix1,iy0] += (hx)*(1-hy)/nv[ix1][iy0]
        G[ix0,iy1] += (1-hx)*(hy)/nv[ix0][iy1]
        G[ix1,iy1] += (hx)*(hy)/nv[ix1][iy1]

#@jit(nopython=True)
def Weighting(xx,yy,nv,N):
    G = np.zeros((nz+1,nr+1))
    for i in range (0,N):
        hx0 = xx[i]
        hy0 = yy[i]
        ix0 = int(hx0/dz)
        ix1 = ix0 + 1
        iy0 = int(hy0/dr)
        iy1 = iy0 + 1
        hx = abs(int(hx0/dz) - (hx0/dz))
        hy = abs(int(hy0/dr) - (hy0/dr))
        G[ix0,iy0] += (charge*(1-hx)*(1-hy))#/nv[ix0][iy0]
        G[ix1,iy0] += (charge*(hx)*(1-hy))#/nv[ix1][iy0]
        G[ix0,iy1] += (charge*(1-hx)*(hy))#/nv[ix0][iy1]
        G[ix1,iy1] += (charge*(hx)*(hy))#/nv[ix1][iy1]
    print(G)
    G = G/nv
    #print(np.sum(G))
    return G

nodalvolume(nodal_volume)
print(nodal_volume)
#weighting(grid,X,Y,nodal_volume)
#rho = Weighting(Pz,Pr,nodal_volume,N)
#print(np.sum(nodal_volume))
#print(np.sum(grid))
#print(np.sum(rho))
