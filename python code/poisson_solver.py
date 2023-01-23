import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from mpl_toolkits.mplot3d import Axes3D
import drawnow

z = 2
r = 1
dz = 0.01
dr = 0.01
nz = int(z/dz)
nr = int(r/dr)
epsi0 = 1
charge = 1
mass = 1
grid = dz*(np.mgrid[0:(z/dz)+3,0:(r/dr)+3])
w = 1.9

r_arr = np.linspace(0,r,num=nr+1)
#print(r_arr)
rho = np.zeros((nz+1,nr+1))
phi1 = np.zeros((nz+3,nr+3))
phi11 = np.zeros((nz+3,nr+3))
elecZ = np.zeros((nz+1,nr+1))
elecR = np.zeros((nz+1,nr+1))
nodal_volume = np.ones((nz+1,nr+1))
Pz = [1.0]
Pr = [0.5]
N = len(Pz)

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
    #print(G)
    G = G/nv
    #print(np.sum(G))
    return G

@jit(nopython = True)   
def feild_solve_rz(rho,phi1,r_arr):
    phi1[1,:] = phi1[2,:]
    #for i in range (int(4*nr/10),int(6*nr/10)):
    #    phi1[1,i] = 5000
    phi1[nz+1,:] = phi1[nz,:]
    phi1[:,1] = phi1[:,2]
    phi1[:,nr+1] = phi1[:,nr]
    for i in range (2,nz+1):
        for j in range (2,nr+1):
            if (i+j)%2 == 0:
                prev = phi1[i,j]
                t1 = rho[i,j]/epsi0
                t2 = (phi1[i][j+1] + phi1[i][j-1])/(dr**2)
                t3 = (phi1[i][j+1] - phi1[i][j-1])/(2.0*dr*dr*(j-1))
                t4 = (phi1[i-1][j]+phi1[i+1][j])/(dz**2)
                t5 = (2.0/(dr**2)) + (2.0/(dz**2))
                phi1[i,j] = prev + w*(((t1+t2+t4+t3)/t5)-prev)
            else:
                pass

    for i in range (2,nz+1):
        for j in range (2,nr+1):
            if (i+j)%2 != 0:
                prev = phi1[i,j]
                t1 = rho[i,j]/epsi0
                t2 = (phi1[i][j+1] + phi1[i][j-1])/(dr**2)
                t3 = (phi1[i][j+1] - phi1[i][j-1])/(2.0*dr*dr*(j-1))
                t4 = (phi1[i-1][j]+phi1[i+1][j])/(dz**2)
                t5 = (2.0/(dr**2)) + (2.0/(dz**2))
                phi1[i,j] = prev + w*(((t1+t2+t4+t3)/t5)-prev)
            else:
                pass

    phi1[0,:] = 2*phi1[1,:] - phi1[2,:]
    phi1[nz+2,:] = 2*phi1[nz+1,:] - phi1[nz,:]
    phi1[:,0] = 2*phi1[:,1] - phi1[:,2]
    phi1[:,nr+2] = 2*phi1[:,nr+1] - phi1[:,nr]

@jit(nopython = True)   
def feild_solve_rz1(rho,phi11,r):
    #phi11[1,:] = 1000#phi11[2,:]
    for i in range (int(4*nr/10),int(6*nr/10)):
        phi11[1,i] = 5000
    phi11[nz+1,:] = 0#phi11[nz,:]
    phi11[:,1] = 0#phi11[:,2]
    phi11[:,nr+1] = 0#phi11[:,nr]
    for i in range (2,nz+1):
        for j in range (2,nr+1):
            t1 = rho[i,j]/epsi0
            t2 = (phi11[i][j+1] + phi11[i][j-1])/(dr**2)
            t3 = (phi11[i][j+1] - phi11[i][j-1])/(2.0*dr*r[j-1])
            t4 = (phi11[i-1][j]+phi11[i+1][j])/(dz**2)
            t5 = (2.0/(dr**2)) + (2.0/(dz**2))
            phi11[i,j] = (t1+t2+t3+t4)/t5
            

    phi11[0,:] = 2*phi11[1,:] - phi11[2,:]
    phi11[nz+2,:] = 2*phi11[nz+1,:] - phi11[nz,:]
    phi11[:,0] = 2*phi11[:,1] - phi11[:,2]
    phi11[:,nr+2] = 2*phi11[:,nr+1] - phi11[:,nr]

@jit(nopython = True) 
def elec(phi1,elecZ,elecR):
    for i in range (0,nz+1):
        for j in range (0,nr+1):
            elecZ[i,j] = (phi1[i,j] - phi1[i+2,j])/(2.0*dz)
            elecR[i,j] = (phi1[i,j] - phi1[i,j+2])/(2.0*dr)
    return(elecZ,elecR)

def plot1():
    plt.figure(2)
    plt.plot(r,phi1[1,:])
    plt.draw()
    plt.pause(0.01)
    plt.clf()

#feild_solve_rz(rho,phi1,r)
T = 500
t = np.zeros((101))
val = np.zeros_like(t)
zz = np.linspace(0,z,nz+3)
zr = np.linspace(0,r,nr+3)
nodalvolume(nodal_volume)
rho = Weighting(Pz,Pr,nodal_volume,N)
for i in range (0,T):
    #prev_poten = phi1
    feild_solve_rz(rho,phi1,r_arr)
    #print(phi1)
    #feild_solve_rz1(rho,phi11,r_arr)
    #print(np.sum(prev_poten))
    #if (i%200 == 0):
    #    k = int(i/300)
    #    val[k] = (np.sum(phi1)-np.sum(prev_poten))
    #    t[k] = i
    #plt.figure(0)
    #plt.plot(zz,phi1[:,6])
    #plt.draw()
    #plt.pause(0.01)
    #plt.clf()
    #plt.figure(1)
    #plt.plot(zz,phi11[:,6])
    #plt.draw()
    #plt.pause(0.01)
    #plt.clf()
#print(phi1[:,6],phi11[:,6])
#plt.plot(zz,phi1[:,6])
#plt.plot(zz,phi11[:,6])
#plt.show()
#elecZ,elecR = elec(phi1,elecZ,elecR)

#print(elecZ)

#print(zz)
#plt.figure()
#plt.plot(zz,phi1[1,:])
#plt.figure(2)
#plt.plot(t,val)
#plt.ylim(149000,150000)
#print(np.max(phi1))
breaks = np.linspace(0,1.09,1000)
breaks1 = np.linspace(0,1.09,10)
plt.figure(1)
plt.contourf(grid[0],grid[1],phi1,breaks,cmap = 'seismic')
plt.colorbar(ticks=breaks1,orientation="vertical")
plt.title("Potential Feild around a point charge")
plt.xlim(-0.2,2.2)
plt.ylim(-0.2,1.2)
plt.xlabel("Z")
plt.ylabel("R")
#plt.annotate("5000 V", xy= (-0,-0.1))
#plt.annotate("0 V", xy= (2,-0.1))
#plt.figure(2)
#plt.contourf(grid[0],grid[1],phi11,breaks,cmap = 'gnuplot')
plt.show()