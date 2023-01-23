from scipy.stats import maxwell
from initialization import const_density_pl
from initialization import maxwelian_v_dist

def source(xlo,xr,ylo,yu,vt,Px,Py,Vx,Vy):
    new_px,new_py,N = const_density_pl(xlo,xr,ylo,yu)
    new_vx,new_vy = maxwelian_v_dist(vt,N)
    Px = np.concatenate((Px, new_px), axis=None)
    Py = np.concatenate((Px, new_py), axis=None)
    Vx = np.concatenate((Vx, new_vx), axis=None)
    Vy = np.concatenate((Vx, new_vy), axis=None)
    return Px,Py,Vx,Vy

def sink (xl,xr,yl,yu,Px,Py):
    indices = []
    for i in range (0,len(Px)):
        x = Px[i]
        y = Py[i]
        if x>xl and x<xr and y>yl and y<yu:
            indices.append(i)    
    new_Px = np.delete(Px,indices)
    new_Py = np.delete(Py,indices)      
    return new_Px,new_Py

def less_mirror(b,arr,vel):
    for i in range (0,len(arr)):
        loc = arr[i]
        if loc > b:
            arr[i] = b - (loc-b)
            vel[i] = -1*vel[i]

def greater_mirror(b,arr):
    for i in range (0,len(arr)):
        loc = arr[i]
        if loc < b:
            arr[i] = b - (loc-b)
            vel[i] = -1*vel[i]