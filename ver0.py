import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import scipy.stats as stats
from numba import jit
import sys
import time
import drawnow
from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
    AdaptiveETA, FileTransferSpeed, FormatLabel, Percentage, \
    ProgressBar, ReverseBar, RotatingMarker, \
    SimpleProgress, Timer


#constants
pi = np.pi
vb = 4.0
vth = 1.0
epsilon0 = 1.0
e = 1.0/20
m = 1.0/20

#domain setup
L = 100
N =10000
M = 1000
dx = (L*1.0)/M
X = np.linspace(0,L,num=M,endpoint=False)
rho = np.zeros_like((X))
n0 = (2.0*N)/M
wp =  ((2*((N*1.0)/M)*(e**2))/(epsilon0*m))**0.5     #plasma frequency
db = (1.0*vth)/wp                                    #debye length

t0 = 0.0
tf = 50.0
#dt = 0.1
dt = (1.0)/(wp*10)
print('time step = '+ str(dt))
T = int((tf-t0)/dt)

xp = np.zeros((T))
vp = np.zeros((T))
tep = np.zeros((T))
tp = np.zeros((T))
tp0 = np.linspace(0,50, num = 499)
print(np.shape(tp0))
Y = np.linspace(0,L,num=M,endpoint=False)

np.random.seed(4744)
pxl = L*(np.random.rand(1,N))
np.random.seed(3125)
pxr = L*(np.random.rand(1,N))

mesh_charge = np.zeros((T))
total_charge = np.zeros((T))
total_Elec = np.zeros((T))
total_Poten = np.zeros((T))
total_area = np.zeros((T))
total_coeff = np.zeros((int((M*0.5)+1),T),dtype = complex)
#print(np.shape(total_coeff))

dx1 = 0.001
O = int((L*1.0)/0.001)
xa = np.linspace(dx1,L-dx1,num=O-1)
P = len(xa)

print("plasma freq. = " + str(wp))
print("debye length = " + str(db))

@jit(nopython=True)
def periodic_bc(l,r):
    for i in range(0,N):
        if l[i] > L:
            l[i] = (l[i] % L)
            #l[i] -= L
        if l[i] < 0:
            l[i] = (L - (abs(l[i]) % L))
            #l[i] += L
        if r[i] > L:
            r[i] = (r[i] % L)
            #r[i] -= L
        if r[i] < 0:
            r[i] = (L - (abs(r[i]) % L))
            #r[i] += L
    return l,r

@jit(nopython=True)
def particle_density(lx,rx):
    r = np.concatenate((lx,rx))
    G = np.zeros((M))
    for i in range (0,int(2*N)):
        x = r[i]/dx
        m1 = int(x)
        m2 = m1+1
        if x > (L-dx)*10:
            if x == L*10:
                G[0] += (abs(m2-x))#/(dx*dx)
                G[M-1] += (abs(m1-x))#/(dx*dx)
                #print(x,m1,m2,G[0],G[M-1],'a1')
            else:
                G[0] += (abs(m1-x))#/(dx*dx)
                G[M-1] += (abs(m2-x))#/(dx*dx)
                #print(x,m1,m2,G[0],G[M-1],'a2')
        else:
            if x == 0:
                G[m1] += (abs(m2-x))#/(dx*dx)
                G[m2] += (abs(m1-x))#/(dx*dx)
                #print(x,m1,m2,G[m1],G[m2],'a3')
            else:
                G[m2] += (abs(m1-x))#/(dx*10)
                G[m1] += (abs(m2-x))#/(dx*10)
                #print(x,m1,m2,G[m1],G[m2],'a4')
    return G

def field(r):
    r_spec0 = np.fft.rfft(r,M)
    Q = len(r_spec0)
    #print(Q)
    #print(r_spec0)
    p_spec = np.zeros((Q),dtype=complex)
    e_spec = np.zeros((Q),dtype=complex)

    for i in range (1,Q):
        k = (2.0*pi*i)/L
        theta1 = k*dx
        K2 = k*(np.sin(theta1)/theta1)
        theta = theta1*0.5
        K1 = k*(np.sin(theta)/theta)
        
        p_spec[i] = r_spec0[i]/(K1**2)
        e_spec[i] = (1j*K2*p_spec[i])

    pf = np.fft.irfft(p_spec,M)
    ef = np.fft.irfft(e_spec,M)
    return ef,pf,e_spec

@jit(nopython=True)
def interpolate(l,r,G):
    X2 = np.zeros((N))
    X1 = np.zeros((N))
    lx = l*10.0
    rx = r*10.0
    for k in range (0,N):
        l0 = lx[k]
        if l0 > ((L-dx)*10):
            m1 = int(l0)
            m2 = int(l0+1)
            X1[k] = (G[0]*((abs(m1-l0))/(dx*10.0))) + (G[M-1]*((abs(m2-l0))/(dx*10.0)))
            #print(l0,m1,m2,G[M-1],G[0],X1[k],"a1")
        else:
            m1 = int(l0)
            m2 = int(l0+1)
            X1[k] = (G[m2]*((abs(m1-l0))/(dx*10.0))) + (G[m1]*((abs(m2-l0))/(dx*10.0)))
            #print(l0,m1,m2,G[m1],G[m2],X1[k],"a2")

        r0 = rx[k]
        if r0 > ((L-dx)*10):
            n1 = int(r0)
            n2 = int(r0+1)
            X2[k] = (G[0]*((abs(n1-r0))/(dx*10.0))) + (G[M-1]*((abs(n2-r0))/(dx*10.0)))
            #print(r0,n1,n2,G[M-1],G[0],X2[k],"a3")
        else:
            n1 = int(r0)
            n2 = int(r0+1)
            X2[k] = (G[n2]*((abs(n1-r0))/(dx*10.0))) + (G[n1]*((abs(n2-r0))/(dx*10.0)))
            #print(r0,n1,n2,G[n1],G[n2],X2[k],"a4")
    return X1,X2

#@jit(nopython=True)
def interpolate1(l,G):
    X1 = np.zeros((P))
    lx = l*10.0
    for k in range (0,P-1):
        l0 = lx[k]
        if l0 > ((L-dx1)/dx1):
            m1 = int(l0)
            m2 = int(l0+1)
            X1[k] = (G[0]*((abs(m1-l0))/(dx1/dx1))) + (G[M-1]*((abs(m2-l0))/(dx1/dx1)))
            #print(l0,m1,m2,G[M-1],G[0],X1[k],"a1")
        else:
            m1 = int(l0)
            m2 = int(l0+1)
            X1[k] = (G[m2]*((abs(m1-l0))/(dx1/dx1))) + (G[m1]*((abs(m2-l0))/(dx1/dx1)))
            #print(l0,m1,m2,G[m1],G[m2],X1[k],"a2")
    return X1


#@jit(nopython=True)
def getE(A,B):
    A1,B1 = periodic_bc(A,B)
    C1 = particle_density(A1,B1)
    C2 = (C1/20) - 1.0
    D1,D2,coefft = field(C2)
    F1,F2 = interpolate(A1,B1,D1)
    return F1,F2,D1,D2,C1,coefft,C2

@jit(nopython=True)
def changes(a1,a2,b1,b2):

    a01 = a1*dt
    a02 = a2*dt
    a03 = -1.0*b1*dt
    a04 = -1.0*b2*dt
    return a01,a02,a03,a04

@jit(nopython=True)
def new(a3,a4,a5,a6,a7,a8,a9,a10,Z):

    a11 = a3 + (a7*Z)
    a12 = a4 + (a8*Z)
    a13 = a5 + (a9*Z)
    a14 = a6 + (a10*Z)
    return a11,a12,a13,a14

@jit(nopython=True)
def total(h1,h2,h3,h4):
    return ((h1 + (2.0*h2) + (2.0*h3) + h4)/6.0)


def Rk4(l,r,lv,rv,G1,G2):
    #dxl1 = lv*dt
    #dxr1 = rv*dt
    #dvl1 = G1*dt
    #dvr1 = G2*dt
    dxl1,dxr1,dvl1,dvr1 = changes(lv,rv,G1,G2)
    #print(np.shape(dxr1),np.shape(r))
    #xl_dxl1 = l + (dxl1*0.5)
    #xr_dxr1 = r + (dxr1*0.5)
    #vl_dvl1 = lv + (dvl1*0.5)
    #vr_dvr1 = rv + (dvr1*0.5)
    xl_dxl1,xr_dxr1,vl_dvl1,vr_dvr1 = new(l,r,lv,rv,dxl1,dxr1,dvl1,dvr1,0.5)
    G11,G12,D1,D2,C1,D3,C2 = getE(xl_dxl1,xr_dxr1)
    #dxl2 = vl_dvl1*dt
    #dxr2 = vr_dvr1*dt
    #dvl2 = G11*dt
    #dvr2 = G12*dt
    dxl2,dxr2,dvl2,dvr2 = changes(vl_dvl1,vr_dvr1,G11,G12)
    #xl_dxl2 = l + (dxl2*0.5)
    #xr_dxr2 = r + (dxr2*0.5)
    #vl_dvl2 = lv + (dvl2*0.5)
    #vr_dvr2 = rv + (dvr2*0.5)
    xl_dxl2,xr_dxr2,vl_dvl2,vr_dvr2 = new(l,r,lv,rv,dxl2,dxr2,dvl2,dvr2,0.5)
    G13,G14,D1,D2,C1,D3,C2 = getE(xl_dxl2,xr_dxr2)
    #dxl3 = vl_dvl2*dt
    #dxr3 = vr_dvr2*dt
    #dvl3 = G13*dt
    #dvr3 = G14*dt
    dxl3,dxr3,dvl3,dvr3 = changes(vl_dvl2,vr_dvr2,G13,G14)
    #xl_dxl3 = l + dxl3
    #xr_dxr3 = r + dxr3
    #vl_dvl3 = lv + dvl3
    #vr_dvr3 = rv + dvr3
    xl_dxl3,xr_dxr3,vl_dvl3,vr_dvr3 = new(l,r,lv,rv,dxl3,dxr3,dvl3,dvr3,1.0)
    G15,G16,D1,D2,C1,D3,C2 = getE(xl_dxl3,xr_dxr3)
    #dxl4 = vl_dvl3*dt
    #dxr4 = vr_dvr3*dt
    #dvl4 = G15*dt
    #dvr4 = G16*dt
    dxl4,dxr4,dvl4,dvr4 = changes(vl_dvl3,vr_dvr3,G15,G16)
    #dvlf = (dvl1 + (2.0*dvl2) + (2.0*dvl3) + dvl4)/6.0
    #dvrf = (dvr1 + (2.0*dvr2) + (2.0*dvr3) + dvr4)/6.0
    #dxlf = (dxl1 + (2.0*dxl2) + (2.0*dxl3) + dxl4)/6.0
    #dxrf = (dxr1 + (2.0*dxr2) + (2.0*dxr3) + dxr4)/6.0
    dvlf = total(dvl1,dvl2,dvl3,dvl4)
    dvrf = total(dvr1,dvr2,dvr3,dvr4)
    dxlf = total(dxl1,dxl2,dxl3,dxl4)
    dxrf = total(dxr1,dxr2,dxr3,dxr4)
    lvf = lv + dvlf
    rvf = rv + dvrf
    lf = l + dxlf
    rf = r + dxrf
    #print(dvl1,dvl2,dvl3,dvl4,dvlf)
    return lf,rf,lvf,rvf

#@jit(nopython=True)
def area(e1,xx):
    e2 = interpolate1(xx,e1)
    e3 = np.absolute(e2)
    vl = np.sum(e3)
    ar = np.trapz(e3,dx=dx1)
    return ar

@jit(nopython=True)
def errorfun(vl,vr,Rho,poten,vp,tep,tp,j):
    v0 = (np.sum(vl) + np.sum(vr))#/(2*N))
    pe = 0.5*np.sum(Rho*poten)
    ke = m*0.5*(np.sum(vl**2)+np.sum(vr**2))
    te = pe+ke
    tep[j] = (te-te0)/te0
    vp[j] = v0
    tp[j] = j

def growth_rate(nu):
    mag_basis = np.zeros((nu,T))
    #diff_mag_basis = np.zeros((nu,T-1))
    for k in range (0,T):
        #coeff = np.zeros((int((M*0.5)+1)),dtype = complex)
        for ii in range (0,nu+1):
            coeff = np.zeros((int((M*0.5)+1)),dtype = complex)
            coeff[ii] = total_coeff[ii,k]
            wave = np.fft.irfft(coeff,M)
            #mag_basis[ii-1,k] = (np.trapz(abs(wave),dx=0.1))/(4.0*ii)
            mag_basis[ii-1,k] = (np.sum((wave/1000)**2))

    #for k in range (0,nu):
        #for i in range (0,T-1): 
            #diff_mag_basis[k,i] = (mag_basis[k,i+1]-mag_basis[k,i])/dt
    return mag_basis,wave


def power_growth_rate1():
    tc = (np.absolute(total_coeff))**2
    avg_p = (np.sum(tc,axis=0)/(500))
    avg_p_g = np.diff(avg_p,n=1)
    plt.figure(12)
    t = np.linspace(0,tf,num=(len(avg_p)))
    plt.plot(t,avg_p)
    

def power_growth_rate():
    for k in range (2,3):
        tc = total_coeff[k,:]
        print(tc)
        avg_p = np.log((np.absolute(tc))/2.0)
    plt.figure(12)
    t = np.linspace(0,tf,num=(len(avg_p)))
    plt.plot(t,avg_p)
    

def phase_plot(j,xl,xr,vl,vr):
    if (j%10.0) == 0.0:
        fig = plt.figure()
        pos=np.concatenate((xl,xr),axis=0)
        vel=np.concatenate((vl,vr),axis=0)
        plt.scatter(pos,vel,s=1)
        plt.ylim(-10,10)
        plt.xlim(0,100)
        plt.xlabel('X')
        plt.ylabel('V')
        ti = j*dt
        plt.title('Two_Stream_Instability' + str(ti))
        plt.savefig('graph_{}.png'.format(j),format="PNG")
        plt.close(fig)

def te_plot():
    plt.figure(1)
    plt.plot(tp,tep)
    plt.title('error plot TE')
    plt.savefig("te.png")

def v_plot():
    plt.figure(2)
    plt.plot(tp,vp)
    plt.title('error plot V')
    plt.savefig("v.png")

def tc_plot():
    plt.figure(5)
    plt.plot(tp,total_charge)
    plt.title('error plot total charge')
    plt.savefig("tc.png")

def telec_plot():
    plt.figure(6)
    plt.plot(tp,total_Elec)
    plt.title('error plot total Electric')
    plt.savefig("telec.png")

def tpoten_plot():
    plt.figure(7)
    plt.plot(tp,total_area)
    plt.title('error plot total Poten')
    plt.savefig("tpoten.png")

def coeff_plot():
    plt.figure(8)
    plt.plot(tp,np.real(total_coeff).T)
    plt.figure(9)
    plt.plot(tp,np.imag(total_coeff).T)
    plt.title('coeff')

def growth_plot(k):
    for i in range (0,k):
        plt.figure(i)
        plt.title('gr')
    #print(np.shape(tp0))
        a = Diff_magnitude_basis[i,:]
        plt.plot(tp,np.log(a))
    #plt.ylim(-0.9,0.9)
    #plt.draw()
    #plt.pause(0.01)
    #plt.clf()
    #plt.clf()
 
def figdraw(H,R,a):
    plt.plot(H,R)
    plt.ylim(-a,a)
    plt.draw()
    plt.pause(0.01)
    plt.clf()

pbar = ProgressBar(widgets=[Percentage(), Bar(),ETA()], maxval=T).start()


data = np.loadtxt('phase0.out')
#data1 = np.loadtxt ('phase-001.out')
#data_den = np.loadtxt ('density-000.out')
#data_elec = np.loadtxt ('electric-000.out')
Xf = data[:,0]
Vf = data[:,1]

N0 = len(Vf)
u0 = np.sum(Vf)
xx = np.split(Xf,2)
vv = np.split(Vf,2)
#print(vv)
xl = xx[0]
xr = xx[1]
vl = vv[0]
vr = vv[1]
#vl = 3.0*np.ones_like(xl)
#vr = -3.0*np.ones_like(xr)
print(len(xl))

E1 = np.zeros_like((vl))
E2 = np.zeros_like((E1))

E1,E2,elec,poten,den,Coefft,Rho = getE(xl,xr)
pe0 = 0.5*np.sum(Rho*poten)
ke0 = m*0.5*(np.sum(vl**2)+np.sum(vr**2))
te0 = pe0+ke0
print('initial velocity = ' + str(u0))
def main(xl,xr,vl,vr):
    for j in range (0,T):
        phase_plot(j,xl,xr,vl,vr)
        E1,E2,elec,poten,den,Coefft,Rho = getE(xl,xr)
        mesh_charge[j] = np.sum(Rho)
        total_charge[j] = np.sum(den)
        total_Elec[j] = ((np.sum(elec**2))*0.0005)
        #total_Elec[j] = (np.log(np.sum(elec**2)))/2.0
        total_Poten[j] = np.sum(poten)
        total_coeff[:,j] = Coefft 
        #total_area[j] = area(elec,xa)
        xl,xr,vl,vr = Rk4(xl,xr,vl,vr,E1,E2)
        xl,xr = periodic_bc(xl,xr)
        errorfun(vl,vr,Rho,poten,vp,tep,tp,j)
        pbar.update(j+1)

main(xl,xr,vl,vr)
pbar.finish()
#print(total_coeff)
#power_growth_rate()
telec_plot()
#coeff_plot()
Diff_magnitude_basis,Wave = growth_rate(10)
#final_gr = (np.sum(Diff_magnitude_basis,axis = 1))/500
#print(final_gr)
#print(Diff_magnitude_basis)
growth_plot(10)
plt.show()