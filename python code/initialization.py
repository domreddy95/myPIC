import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import drawnow
from scipy.stats import maxwell


p_rho = 100000
z = 2
r = 1
dz = 0.1
dr = 0.1
nz = int(z/dz)
nr = int(r/dr)
nodal_volume = np.zeros((nz+1,nr+1))
vt = 10
vb = 50

def cell_volume(yl,yu):
    ny = round(abs(yu-yl)/dr)
    vol = np.zeros((ny))
    for i in range (0,ny):
        r0 = dr*i
        r1 = dr*(i+1)
        vol[i] = ((r1**2)-(r0**2))*dz
    return vol


#@jit(nopython=True)
def const_density_pl(xlo,xr,ylo,yu):
    nx = int(abs(xlo-xr)/dz)
    ny = round(abs(yu-ylo)/dr)

    nv = cell_volume(ylo,yu)
    
    for i in range (0,nx):
        for j in range (0,ny):
            Np = int(p_rho*nv[j])
            xl = xlo + (dz*i)
            yl = ylo + (dr*j)
            px = xl+(dz*(np.random.rand(Np)))
            py = yl+(dr*(np.random.rand(Np)))
            if i+j == 0:
                Px = px
                Py = py
            else:
                Px = np.concatenate((Px, px), axis=None)
                Py = np.concatenate((Py, py), axis=None)

    error = int((p_rho*((yu**2)-(ylo**2))*abs(xlo-xr)) - (len(Px)))
    #print((p_rho*((yu**2)-(yl**2))*abs(xl-xr)))
    px = abs(xlo-xr)*(np.random.rand(error))
    py = abs(yu-ylo)*(np.random.rand(error))
    Px = np.concatenate((Px, px), axis=None)
    Py = np.concatenate((Py, py), axis=None)

    return Px,Py,len(Px)

@jit(nopython=True)
def angles(cosi,sini,N):
    i = 0
    np.random.seed(1056)
    while (i < N):
        r1 = np.random.rand()
        r2 = np.random.rand()
        r3 = np.random.rand()
        #print(r3)
        if r1**2 + r2**2 < 1:
            cosi[i] = (r1**2 - r2**2)/(r1**2 + r2**2)
            if r3 < 0.5:
                sini[i] = (2.0*r1*r2)/(r1**2 + r2**2)
            else:
                sini[i] = (-2.0*r1*r2)/(r1**2 + r2**2)
            i  += 1
    return cosi,sini

def maxwelian_v_dist(vt,vb,N):
    vel = maxwell.rvs(loc=vt,scale=vt,size=int(N))
    #plt.figure(3)
    #plt.hist(vel,bins=100)
    cosi = np.zeros(N)
    sini = np.zeros(N)
    cosi,sini = angles(cosi,sini,N)
    return (vel*cosi+vb),vel*sini

Px,Py,N = const_density_pl(0,z,0,r)
Vx,Vy = maxwelian_v_dist(vt,vb,N)
plt.figure(1)
plt.scatter(Px,Py,s=1)
plt.figure(2)
plt.scatter(Vx,Vy,s=1)
plt.figure(4)
plt.hist(Vx,bins=100)
plt.figure(5)
plt.hist(Vy,bins=100)
plt.show()