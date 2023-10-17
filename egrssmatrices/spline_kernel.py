import numpy as np

def spline_kernel(t,p):
    T = np.vander(t, 2*p)/np.flip(np.cumprod(np.hstack([1,np.arange(1,2*p)])))
    Ut = T[:,p:].T
    Vt = (np.fliplr(T[:,:p])*((-1)**np.arange(0,p))).T
    return Ut,Vt
