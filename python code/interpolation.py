import numpy as np
import matplotlib.pyplot as plt

@jit(nopython=True)
def interpolate(Ex,Ey,xx,yy,eex,eey):
    for i in range (0,N):
        hx = xx[i]
        hy = yy[i]
        ix0 = int(hx/dz)
        ix1 = ix0 + 1
        iy0 = int(hy/dr)
        iy1 = iy0 + 1
        xv0 = (1-hx)*(1-hy)*Ex[ix0][iy0]
        xv1 = (hx)*(1-hy)*Ex[ix1][iy0]
        xv2 = (1-hx)*(hy)*Ex[ix0][iy1]
        xv3 = (hx)*(hy)*Ex[ix1][iy1]
        eex[i] = xv0+xv1+xv2+xv3
        yv0 = (1-hx)*(1-hy)*Ey[ix0][iy0]
        yv1 = (hx)*(1-hy)*Ey[ix1][iy0]
        yv2 = (1-hx)*(hy)*Ey[ix0][iy1]
        yv3 = (hx)*(hy)*Ey[ix1][iy1]
        eey[i] = yv0+yv1+yv2+yv3
    