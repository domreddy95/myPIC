import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from numba import jit
import drawnow
from scipy.stats import maxwell
from mpl_toolkits import mplot3d
from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
    AdaptiveETA, FileTransferSpeed, FormatLabel, Percentage, \
    ProgressBar, ReverseBar, RotatingMarker, \
    SimpleProgress, Timer

charge = 1
e = 1
m = 1
mass = 1

t0 = 0
dt = 0.00001
T = int((4.25*2*np.pi)/1/dt)+1
print(T)

Ez = 0
Er = 0
Et = 0
Br = 0
Bz = 0
Bt = (np.linspace(0,100,num=T))
#B = 1
#print(Bt)


Z = np.zeros((T))-1
R = np.zeros((T))
Z1 = np.zeros((T))
R1 = np.zeros((T))
theta = np.zeros((T))
Uz = np.zeros((T))
Vr = np.zeros((T))
Vt = np.zeros((T))
Uz[0] = 0
Vr[0] = 1
Vt[0] = 0
Uz1 = np.zeros((T))
Vr1 = np.zeros((T))
Vt1 = np.zeros((T))
Uz1[0] = 1
Vr1[0] = 0
Vt1[0] = 0


#@jit(nopython=True)
def boris_pusher0(zi,ri,ti,uz0,ur0,ut0,Ez,Er,Et,Bz,Br,Bt):
    uz1 = uz0 + (charge*Ez*dt)/(2*mass)
    ur1 = ur0 + (charge*Er*dt)/(2*mass)
    ut1 = ut0 + (charge*Et*dt)/(2*mass)
    a = np.array([uz1,ur1,ut1])
    #(a)
    b = np.array([Bz,Br,Bt])
    #(b)
    t = (charge*b*dt)/(2*m)
    #(t)
    a1 = a + np.cross(a,b)
    #(a1)
    s = (2*t)/(1+(np.linalg.norm(t)**2))
    #(s)
    z = a + np.cross(a1,s)
    #(z)
    uz2 = z[0] + (charge*Ez*dt)/(2*mass)
    ur2 = z[1] + (charge*Er*dt)/(2*mass)
    ut2 = z[2] + (charge*Et*dt)/(2*mass)
    zf = zi + uz2*dt
    rf = ri + ur2*dt
    tf = ti + ut2*dt
    return zf,rf,tf,uz2,ur2,ut2

@jit(nopython=True)
def boris_pusher(zi,ri,ti,uz0,ur0,ut0,Ez,Er,Et,Bz,Br,Bt,c,s):
    uz1 = uz0 + (charge*Ez*dt)/(2*mass)
    ur1 = ur0 + (charge*Er*dt)/(2*mass)
    #ut1 = 0#ut0 + (charge*Et*dt)/(2*mass)
    az1 = c*uz1 + s*ur1
    ar1 = -s*uz1 + c*ur1
    uz2 = az1 + (charge*Ez*dt)/(2*mass)
    ur2 = ar1 + (charge*Er*dt)/(2*mass)
    ut2 = 0#z[2] + (charge*Et*dt)/(2*mass)
    zf = zi + uz2*dt
    rf = ri + ur2*dt
    tf = 0#ti + ut2*dt
    return zf,rf,tf,uz2,ur2,ut2

def boris_pusher1(x,y,u,v,B):
    u0 = u + (e*Ez*dt)/(2*m)
    a = np.array([u0,v,0])
    b = np.array([0,0,B])
    def f(z):
        c = z+a
        return (np.cross(c,b))*((e*dt)/(2*m))-z+a

    z = fsolve(f,[a])
    #print(z)
    U = z[0] + (e*Ez*dt)/(2*m)
    V = z[1]
    X = x + u*dt
    Y = y + v*dt
    return X,Y,U,V



pbar = ProgressBar(widgets=[Percentage(), Bar(),ETA()], maxval=T).start()
for i in range (0,T-1):
    Bt = 1 + 0.01*Z[i]
    #t = (charge*Bt*dt)/(2*m)
    #s = (2*t)/(1+(t**2))
    #c = (1-(t**2))/(1+(t**2))
    #Z[i+1],R[i+1],theta[i+1],Uz[i+1],Vr[i+1],Vt[i+1] = boris_pusher(Z[i],R[i],theta[i],Uz[i],Vr[i],Vt[i],Ez,Er,Et,Bz,Br,Bt,c,s)
    Z1[i+1],R1[i+1],Uz1[i+1],Vr1[i+1]= boris_pusher1(Z1[i],R1[i],Uz1[i],Vr1[i],Bt)
    pbar.update(i+1)

pbar.finish()
#
#plt.figure(1)
#ax = plt.axes(projection='3d')
#ax.plot3D(Z, R, theta, 'gray')
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z')
#ax.view_init(90, 35)
#plt.ylim(-5,5)
#plt.xlim(-5,5)
#plt.show()


B = 10*(np.max(Z)*0.5) + 1
#Tf = int((2*np.pi)/B/dt)
#print(Tf)
##print(10/((B**2)*2))
rd = ((R1[T-1]-R1[int(T/17)])**2)
#zd = (Z[Tf-1]-Z[0])**2
true = 0.5*0.01
error = (np.round(((((rd)**0.5)/((T-(T/17))*dt)))-true,15))/true
print(error,true)
plt.figure(1)
plt.plot(Z1,R1)
plt.ylabel('r')
plt.xlabel('z')
plt.title("gradB drift, implicit plot")
plt.annotate('error = '+ str(error),xy=(0.26,0.7))
#plt.figure()
#plt.plot(Z[int(T/17)],R[int(T/17)],Z[T-1],R[T-1])
#plt.ylim(-5,10)
#plt.xlim(-5,10)

plt.show()
