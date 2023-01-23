import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import drawnow
from scipy.stats import maxwell
from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
    AdaptiveETA, FileTransferSpeed, FormatLabel, Percentage, \
    ProgressBar, ReverseBar, RotatingMarker, \
    SimpleProgress, Timer
from scipy.optimize import fsolve

particle_density = 1E6
charge = 1#/particle_density
mass = 1#particle_density
epsilon = particle_density
thermal_velocity = 1
beam_belocity = 5
plasma_frequency = ((particle_density*(charge**2))/(mass*epsilon))**0.5
debye_length = thermal_velocity/plasma_frequency
#print(plasma_frequency,debye_length)

#mesh_properties
z = 2
r = 1
dz = debye_length/10
dr = debye_length/10
nz = int(z/dz)
nr = int(r/dr)
print(nr)
w = 1.6



#arrays
nodal_volume = np.zeros((nz+1,nr+1))
grid = dz*(np.mgrid[0:int((z/dz)+3),0:int((r/dr)+3)])
#grid = dz*(np.mgrid[0:(z/dz)+3,0:(r/dr)+3])
R = np.linspace(0,r,num=nr+1)
rho = np.zeros((nz+1,nr+1))
grid0 = np.linspace(1,nz+1,num=nz+1)
phi1 = np.zeros((nz+3,nr+3))
grid1 = np.linspace(0,1,num=nr+3)
elecZ = 500*np.ones((nz+1,nr+1))
elecR = np.zeros((nz+1,nr+1))

#time
t0 = 0
dt = 1/(1000*plasma_frequency)
T = 40
tf = T*dt

#plot mesh geometry
Q = 100
linz0 = np.zeros((Q))
linr0 = np.zeros((Q))
linez1 = r*np.ones((Q))
liner1 = z*np.ones((Q))
xaxis = np.linspace(0,z,num=Q)
yaxis = np.linspace(0,r,num=Q)

@jit(nopython=True)
def cell_volume(xlo,xr,ylo,yu):
    #print((abs(yu-ylo)/dr))
    ny = int(abs(yu-ylo)/dr)
    vol = np.zeros((ny))
    for i in range (0,ny):
        r0 = dr*i
        r1 = dr*(i+1)
        vol[i] = ((r1**2)-(r0**2))*(xr-xlo)
    return vol


@jit(nopython=True)
def const_density_pl(xlo,xr,ylo,yu):
    nv = cell_volume(xlo,xr,ylo,yu)
    #print((particle_density*((yu**2)-(ylo**2))*abs(xlo-xr)))
    #nx = int(abs(xlo-xr)/dz)
    ny = round(abs(yu-ylo)/dr)
    #print(nx,ny)
    for i in range (0,1):
        for j in range (0,ny):
            Np = int(particle_density*nv[j])
            xl = xlo
            yl = ylo + (dr*j)
            px = xl+((xr-xlo)*(np.random.rand(Np)))
            py = yl+(dr*(np.random.rand(Np)))
            if i+j == 0:
                Px = px
                Py = py
            else:
                new_Px = np.concatenate((Px, px))
                new_Py = np.concatenate((Py, py))
            Px = new_Px
            Py = new_Py

    error = int((particle_density*((yu**2)-(ylo**2))*abs(xlo-xr)) - (len(Px)))
    #print (error)
    px = xlo + abs(xlo-xr)*(np.random.rand(error))
    py = ylo + abs(yu-ylo)*(np.random.rand(error))
    new_Px = np.concatenate((Px, px))
    new_Py = np.concatenate((Py, py))

    return new_Px,new_Py,len(new_Px)

def maxwelian_v_dist(vt,vb,N):
    mu, sigma = 0,1
    Vx = (np.random.normal(mu,sigma,N))*vt
    Vy = (np.random.normal(mu,sigma,N))*vt
    #Vz = (np.random.normal(mu,sigma,N))*vt
    #V = ((Vx**2)+(Vy**2)+(Vz**2))**0.5
    return Vx,Vy

@jit(nopython=True)
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

@jit(nopython = True) 
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
    #print(np.sum(G))
    G = G/nv
    #print(np.sum(G))
    return G
    

@jit(nopython = True)   
def feild_solve_rz(rho,phi1,r_arr):
    phi1[1,:] = 100#phi1[2,:]
    #for i in range (int(3),int(10)):
        #phi1[1,i] = 100
    phi1[nz+1,:] = 0#phi1[nz,:]
    phi1[:,1] = phi1[:,2]
    phi1[:,nr+1] = phi1[:,nr]
    for i in range (2,nz+1):
        for j in range (2,nr+1):
            if (i+j)%2 == 0:
                prev = phi1[i,j]
                t1 = rho[i,j]/epsilon
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
                t1 = rho[i,j]/epsilon
                t2 = (phi1[i][j+1] + phi1[i][j-1])/(dr**2)
                t3 = (phi1[i][j+1] - phi1[i][j-1])/(2.0*dr*dr*(j-1))
                t4 = (phi1[i-1][j]+phi1[i+1][j])/(dz**2)
                t5 = (2.0/(dr**2)) + (2.0/(dz**2))
                phi1[i,j] = prev + w*(((t1+t2+t4+t3)/t5)-prev)
            else:
                pass
    
    phi11[0,:] = 2*phi11[1,:] - phi11[2,:]
    phi11[nz+2,:] = 2*phi11[nz+1,:] - phi11[nz,:]
    phi11[:,0] = 2*phi11[:,1] - phi11[:,2]
    phi11[:,nr+2] = 2*phi11[:,nr+1] - phi11[:,nr]

    

@jit(nopython = True)
def ghost_cell(phi1):
    phi1[0,:] = 2*phi1[1,:] - phi1[2,:]
    phi1[nz+2,:] = 2*phi1[nz+1,:] - phi1[nz,:]
    phi1[:,0] = 2*phi1[:,1] - phi1[:,2]
    phi1[:,nr+2] = 2*phi1[:,nr+1] - phi1[:,nr]

@jit(nopython = True) 
def elec(phi1,elecZ,elecR):
    for i in range (0,nz+1):
        for j in range (0,nr+1):
            elecZ[i+1,j] = (phi1[i,j] - phi1[i+2,j])/(2.0*dz)
            elecR[i,j+1] = (phi1[i,j] - phi1[i,j+2])/(2.0*dr)

@jit(nopython=True)
def interpolate_elec(Ex,Ey,xx,yy,eex,eey,N):
    for i in range (0,N):
        hx0 = xx[i]
        #print(hx0)
        hy0 = yy[i]
        ix0 = int(hx0/dz)
        #print(ix0)
        ix1 = ix0 + 1
        iy0 = int(hy0/dr)
        iy1 = iy0 + 1
        hx = np.absolute(int(hx0/dz) - (hx0/dz))
        #print(hx)
        hy = np.absolute(int(hy0/dr) - (hy0/dr))
        xv0 = (1-hx)*(1-hy)*Ex[ix0][iy0]
        #print(xv0)
        xv1 = (hx)*(1-hy)*Ex[ix1][iy0]
        xv2 = (1-hx)*(hy)*Ex[ix0][iy1]
        xv3 = (hx)*(hy)*Ex[ix1][iy1]
        eex[i] = xv0+xv1+xv2+xv3
        yv0 = (1-hx)*(1-hy)*Ey[ix0][iy0]
        yv1 = (hx)*(1-hy)*Ey[ix1][iy0]
        yv2 = (1-hx)*(hy)*Ey[ix0][iy1]
        yv3 = (hx)*(hy)*Ey[ix1][iy1]
        eey[i] = yv0+yv1+yv2+yv3

@jit(nopython=True)
def interpolate_poten(Ex,xx,yy,eex,N):
    for i in range (0,N):
        hx0 = xx[i]
        #print(hx0)
        hy0 = yy[i]
        ix0 = int(hx0/dz)+1
        #print(ix0)
        ix1 = ix0 + 1
        iy0 = int(hy0/dr) + 1
        iy1 = iy0 + 1
        hx = abs(int(hx0/dz) - (hx0/dz))
        #print(hx)
        hy = abs(int(hy0/dr) - (hy0/dr))
        #print(Ex[ix1][iy1])
        xv0 = (1-hx)*(1-hy)*Ex[ix0][iy0]
        xv1 = (hx)*(1-hy)*Ex[ix1][iy0]
        xv2 = (1-hx)*(hy)*Ex[ix0][iy1]
        xv3 = (hx)*(hy)*Ex[ix1][iy1]
        #print(xv0,xv1,xv2,xv3)
        eex[i] = xv0+xv1+xv2+xv3

def source(xlo,xr,ylo,yu,Px,Py,Vx,Vy):
    new_px,new_py,N = const_density_pl(xlo,xr,ylo,yu)
    new_vx,new_vy = maxwelian_v_dist(thermal_velocity,beam_belocity,N)
    Px = np.concatenate((Px, new_px), axis=None)
    Py = np.concatenate((Px, new_py), axis=None)
    Vx = np.concatenate((Vx, new_vx), axis=None)
    Vy = np.concatenate((Vx, new_vy), axis=None)
    return Px,Py,Vx,Vy,N

#@jit(nopython=True)
def sink (xlo,xr,ylo,yu,Px,Py,Vx,Vy):
    indices = []
    for i in range (0,len(Px)):
        x = Px[i]
        y = Py[i]
        if x>xlo and x<xr and y>ylo and y<yu:
            indices.append(i)    
    new_Px = np.delete(Px,indices)
    new_Py = np.delete(Py,indices)
    new_Vx = np.delete(Vx,indices)
    new_Vy = np.delete(Vy,indices)     
    return new_Px,new_Py,new_Vx,new_Vy

@jit(nopython=True)
def lesser_mirror(b,arr,vel,N):
    for i in range (0,N):
        loc = arr[i]
        #print(loc)
        if loc > b:
            arr[i] = b - (np.absolute(loc-b))
            vel[i] = -1*vel[i]

@jit(nopython=True)
def greater_mirror(b,arr,vel,N):
    for i in range (0,N):
        loc = arr[i]
        if loc < b:
            arr[i] = b + np.absolute(loc-b)
            vel[i] = -1*vel[i]

@jit(nopython=True)
def push1(zi,ri,ti,uz0,ur0,ut0,Ez,Er,Et,Bz,Br,Bt):
    uz1 = uz0 + (charge*Ez*dt)/(2*mass)
    ur1 = ur0 + (charge*Er*dt)/(2*mass)
    ut1 = ut0 + (charge*Et*dt)/(2*mass)
    a = np.array([uz1,ur1,ut1])
    b = np.array([Bz,Br,Bt])
    return a,b

@jit(nopython=True)
def push2(zi,ri,ti,uz0,ur0,ut0,Ez,Er,Et,Bz,Br,Bt,z):
    uz2 = z[0] + (charge*Ez*dt)/(2*mass)
    ur2 = z[1] + (charge*Er*dt)/(2*mass)
    ut2 = z[2] + (charge*Et*dt)/(2*mass)
    zf = zi + uz2*dt
    #print(zf,zi,uz2)
    rf = ri + ur2*dt
    tf = ti + ut2*dt
    return zf,rf,tf,uz2,ur2,ut2

def boris_pusher(zi,ri,ti,uz0,ur0,ut0,Ez,Er,Et,Bz,Br,Bt):
    a,b = push1(zi,ri,ti,uz0,ur0,ut0,Ez,Er,Et,Bz,Br,Bt)
    #print(a)
    #def f(z):
    #    c = z+a
    #    return ((np.cross(c,b))*((charge*dt)/(2*mass)))-z+a
    z = a#fsolve(f,[a])
    #print('z is' + str(z))
    zf,rf,tf,uz2,ur2,ut2 = push2(zi,ri,ti,uz0,ur0,ut0,Ez,Er,Et,Bz,Br,Bt,z)
    return zf,rf,tf,uz2,ur2,ut2
    

def particle_advection(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,N):
    for i in range (0,N):
        b0,b1,b2,b3,b4,b5=boris_pusher(a0[i],a1[i],a2[i],a3[i],a4[i],a5[i],a6[i],a7[i],a8[i],a9[i],a10[i],a11[i])
        a0[i] = b0
        a1[i] = b1
        a2[i] = b2
        a3[i] = b3
        a4[i] = b4
        a5[i] = b5
    return a0,a1,a2,a3,a4,a5

@jit(nopython=True)
def energy(s_poten,Vx,Vy,N):
    pe = (charge*s_poten)
    ke = (0.5*mass*((Vx**2)+(Vy**2)))
    #print(np.shape(Vx))
    te = pe + ke
    
    return (np.sum(pe))/N,(np.sum(ke))/N,(np.sum(te))/N,

nodalvolume(nodal_volume)
N = 40
Pz = 0.11*np.ones((N))
Pr = np.linspace(0.45,0.55,num=N)
#cellvolume = cell_volume(0.001,0.005,4*dr,6*dr)
#Pz,Pr,N = const_density_pl(0.001,0.05,4*dr,6*dr,cellvolume)
##cellvolume = cell_volume(1.05,1.051,0.25,0.75)
##Pz1,Pr1,N1 = const_density_pl(1.05,1.051,0.25,0.75,cellvolume)
#Pz = Pz0#np.concatenate((Pz0,Pz1))
#Pr = Pr0#np.concatenate((Pr0,Pr1))
#Pz = [0.001]
#Pr = [0.5]
Pt = np.zeros_like(Pz)
#N = 1#len(Pz)
print(N)
#Vz,Vr = maxwelian_v_dist(thermal_velocity,beam_belocity,N)

Vz = 0.01*np.zeros_like(Pz)
Vr = np.zeros_like(Pz)
Vt = np.zeros_like(Pz)
Bz = np.zeros_like(Pz)
Br = np.zeros_like(Pz)
Bt = np.zeros_like(Pz)
p_elecT = np.zeros_like(Pz)

Tt = 260
t0 = np.zeros((Tt))
t1 = np.zeros((Tt))
tt = np.zeros((Tt))
vv = np.zeros((Tt))
TE = np.zeros((Tt))
PE = np.zeros((Tt))
KE = np.zeros((Tt))
E = np.zeros((Tt))

zz = np.linspace(0,z,nz+3)
zr = np.linspace(0,r,nr+3)
Pz_min = beam_belocity*2*((2/np.pi)**0.5)*dt
s_poten = np.zeros((N))
pbar = ProgressBar(widgets=[Percentage(), Bar(),ETA()], maxval=Tt).start()

for i in range (0,75000):
        feild_solve_rz(rho,phi1,R)
for k in range (0,Tt):
    p_elecZ = np.zeros((N))
    p_elecR = np.zeros((N))
    s_poten = np.zeros((N))
    #rho = Weighting(Pz,Pr,nodal_volume,N)
    #print(rho)
    #if k == 0:
    #    for i in range (0,9000000):
    #        feild_solve_rz(rho,phi1,R)
    #        #ghost_cell(phi1)
    #    #plt.plot(zz,phi1[:,6])
    #    #print((phi1[1]-phi1[nz+1])/phi1[11,6])
    #    plt.show()
    #        
    #else:
    #pbar = ProgressBar(widgets=[Percentage(), Bar(),ETA()], maxval=90000000).start()
    #for i in range (0,25000):
        #feild_solve_rz(rho,phi1,R)
        #pbar.update(i+1)

    ghost_cell(phi1)
    #print(phi1)
    #plt.figure(1)
    #plt.contourf(grid[0],grid[1],phi1,cmap = 'seismic')
    #plt.title("Potential Feild around a point charge")
    #plt.xlim(-0.2,2.2)
    #plt.ylim(-0.2,1.2)
    #plt.xlabel("Z")
    #plt.ylabel("R")
    #plt.show()
    elec(phi1,elecZ,elecR)
    interpolate_elec(elecZ,elecR,Pz,Pr,p_elecZ,p_elecR,N)
    #interpolate_poten(phi1,Pz,Pr,s_poten,N)
    #print(p_elecZ,p_elecR)
    #PE[k],KE[k],TE[k] = energy(s_poten,Vz,Vr,N)
    Pz,Pr,Pt,Vz,Vr,Vt = particle_advection(Pz,Pr,Pt,Vz,Vr,Vt,p_elecZ,p_elecR,p_elecT,Bz,Br,Bt,N)
    #print(Pz)
    #print(Pr,Vr)
    if np.amin(Pr)<0:
        greater_mirror(0,Pr,Vr,N)
    if np.amax(Pr)>r:
        lesser_mirror(r,Pr,Vr,N)
    if np.amin(Pz)<0:
        greater_mirror(0,Pz,Vz,N)
    #if np.amax(Pz)>z:
    #    sink(2,100,-10,10,Pz,Pr,Vz,Vr)

    #N = len(Pz)
    #if N == 0:
        #break
    #print(Pz,Pr)
    #print(s_poten)
    #VE[k] = Pz
    #t0[k] = np.sum(Pr)/N
    #tt[k] = Pz[0]
    #vv[k] = Pr[0]

    #print(Vz)
    #print(Vz)
    #print(dt)
    #print(Vz)
    #cellvolume = cell_volume(0.0,Pz_min,0.25,0.75)
    #Pz_s,Pr_s,N_s = const_density_pl(0,Pz_min,0.25,0.75,cellvolume)
    #Vz_s,Vr_s = maxwelian_v_dist(thermal_velocity,beam_belocity,N_s)
    #Pz = np.concatenate((Pz,Pz_s))
    #Pr = np.concatenate((Pr,Pr_s))
    #Vz = np.concatenate((Vz,Vz_s))
    #Vr = np.concatenate((Vr,Vr_s))
    #N += N_s
    #Pt = np.zeros_like(Pz)
    #Vt = np.zeros_like(Pz)
    #Bz = np.zeros_like(Pz)
    #Br = np.zeros_like(Pz)
    #Bt = np.zeros_like(Pz)
    #p_elecT = np.zeros_like(Pz)
    #print(N)
    #cellvolume = cell_volume(0,((vz_min*dt)+dz),0,r)
    #Pz,Pr,Vz,Vr,N = source(0,((vz_min*dt)+dz),0,r,Pz,Pr,Vz,Vr)
    #plt.figure(3)
    #plt.scatter(Pz,Pr,s=1)
    #print(N)
    #Pz,Pr = first_pass_leap_frog(Pz,Pr,Vz,Vr,dt)
    if k%10 == 0:
        plt.figure()
        plt.scatter(Pz,Pr,s=1)
        plt.plot(xaxis,linz0,color='y')
        plt.plot(xaxis,linez1,color='y')
        plt.plot(linr0,yaxis,color='y')
        plt.plot(liner1,yaxis,color='y')
        plt.annotate("no.of particles = " + str(N), xy=(1,-0.15) )
        plt.annotate("100V", xy=(-0.105,-0.15) )
        plt.annotate("0V", xy=(2,-0.15) )
        plt.title("Axisymetric RZ Simulation")
        plt.ylim(-0.25,1.25)
        plt.xlim(-0.25,2.25)
        plt.savefig('1_'+str(k)+'.png', bbox_inches='tight')
        plt.clf()
        #breaks = np.linspace(0,10,1000)
        #breaks1 = np.linspace(0,10,10)
        #plt.figure()
        #plt.contourf(grid[0],grid[1],phi1,cmap = 'seismic')
        #plt.colorbar(ticks=breaks1,orientation="vertical")
        #plt.title("Potential Feild")
        #plt.xlim(-0.2,2.2)
        #plt.ylim(-0.2,1.2)
        #plt.xlabel("Z")
        #plt.ylabel("R")
        #plt.savefig('poten1_'+str(k)+'.png', bbox_inches='tight')
        #pass
    pbar.update(k+1)
    #print(k)
#print(Pz[Tt-1])
#print(KE)
#print()
#print()
pbar.finish()
#print(tt)
#print(PE[0],KE[0],TE[0],Pz[0],Vz[0])
#print(PE[Tt-1],KE[Tt-1],TE[Tt-1],Pz[Tt-1],Vz[Tt-1])
#print((TE[Tt-1]-TE[0])/(TE[0]))

#print(500*(-Pz[0]+2))
#print(s_poten[0])
#plt.figure(3)
#plt.plot(tt,TE/TE[0]-1)
#plt.savefig('TE.png', bbox_inches='tight')
#plt.figure(4)
#plt.scatter(tt,vv,s=1)
#plt.savefig('sumV.png', bbox_inches='tight')
#plt.ylim(0,r)
#plt.xlim(0,z)
#print(t0[Tt-1]-1.5,0.5-t1[Tt-1])
#print(Vz[0],Vz[1])
#plt.show()