import numpy as np

class egrssmatrix:
    def __init__(self,Ut,Vt,d=None):
        self.Ut = Ut
        self.Vt = Vt
        self.p, self.n = Ut.shape
        self.d  = d
        self.Wt = None
        self.c  = None
        self.Yt = None
        self.Zt = None
        self.shape = (self.n, self.n)
        self.dtype = Ut.dtype
        
    def __repr__(self):
        return f"egrssmatrix(p={self.p},n={self.n})"
    
    def __add__(self,other):
        return egrssmatrix(np.hstack((self.Ut,other.Ut)), np.hstack((self.Ut,other.Ut)))
    
    def __matmul__(self,other):
        assert self.n == other.shape[0], f"dimension error. Expected length to be {self.n} got {other.shape[0]}"
        ubar = self.Ut @ other
        vbar = np.zeros(ubar.shape)
        yk = np.zeros(other.shape)
        for k in range(self.n):
            vbar = vbar + self.Vt[:,k]*other[k]
            ubar = ubar - self.Ut[:,k]*other[k]
            yk[k] = np.dot(self.Ut[:,k],vbar) + np.dot(self.Vt[:,k],ubar)
        if self.d is not None:
            yk = yk + self.d*other
        return yk
    
    def __getitem__(self, key):
        if key[0] > key[1]:
            return np.dot(self.Ut[:,key[0]], self.Vt[:,key[1]])
        elif key[0] == key[1]:
            if self.d is not None:
                return np.dot(self.Ut[:,key[0]], self.Vt[:,key[1]]) + self.d[key[0]]
            else:
                return np.dot(self.Ut[:,key[0]], self.Vt[:,key[1]])
        return np.dot(self.Vt[:,key[0]], self.Ut[:,key[1]])
    
    def __setitem__(self,key,value):
        raise IndexError('You can not set index value of an egrssmatrix')
        
    def matvec(self,other):
        return self@other
        
    def full(self):
        K = np.tril(self.Ut.T@self.Vt,-1)
        K = K + K.T + np.diag(np.sum(self.Ut*self.Vt,axis=0))
        if self.d is not None:
            if np.isscalar(self.d):
                K = K + np.diag(self.d*np.ones(self.n))
            else: 
                K = K + np.diag(self.d)
        return K
    
    def cholesky(self):
        wbar = np.zeros((self.p,self.p))
        Wt = self.Vt.copy()
        if self.d is None:
            for k in range(self.n):
                Wt[:,k] = Wt[:,k] - wbar@self.Ut[:,k]
                Wt[:,k] = Wt[:,k]/np.sqrt(np.dot(self.Ut[:,k],Wt[:,k]))
                wbar   = wbar + np.outer(Wt[:,k],Wt[:,k])
            self.Wt = Wt
            self.c = np.sum(self.Ut*self.Wt,axis=0)
        else:
            if np.isscalar(self.d):
                c = self.d*np.ones(self.n)
            else:
                c = self.d.copy()
            
            for k in range(self.n):
                Wt[:,k] = Wt[:,k] - wbar@self.Ut[:,k]
                c[k] = np.sqrt(np.dot(self.Ut[:,k],Wt[:,k]) + c[k])
                Wt[:,k] = Wt[:,k]/c[k]
                wbar   = wbar + np.outer(Wt[:,k],Wt[:,k])
            self.Wt = Wt
            self.c = c
            
    def solve(self,other):
        return self.backward(self.forward(other))
    
    def forward(self,other):
        if self.Wt is None:
            self.cholesky()
        wbar = np.zeros(self.p)
        x = np.zeros(self.n)
        for k in range(self.n):
            x[k] = (other[k] - np.dot(self.Ut[:,k],wbar))/self.c[k]
            wbar = wbar + self.Wt[:,k]*x[k]
        return x
    
    def backward(self,other):
        if self.Wt is None:
            self.cholesky()
        ubar = np.zeros(self.p)
        x = np.zeros(self.n)
        for k in reversed(range(self.n)):
            x[k] = (other[k] - np.dot(self.Wt[:,k],ubar))/self.c[k]
            ubar = ubar + self.Ut[:,k]*x[k]
        return x
    
    def cholesky_inverse(self):
        if self.Yt is not None and self.Zt is not None:
            return
        if self.Wt is None:
            self.cholesky()
        Yt = self.Ut.copy()
        Zt = self.Wt.copy()
        
        for k in range(self.p):
            Yt[k,:] = self.forward(self.Ut[k,:])
            Zt[k,:] = self.backward(self.Wt[k,:])

        Zt = np.linalg.solve(Zt@self.Ut.T - np.eye(self.p), Zt)
        
        self.Yt = Yt
        self.Zt = Zt
        
    def logdet(self):
        if self.c is None:
            self.cholesky()
        return 2*np.sum(np.log(self.c))
    
    def trace_inverse(self):
        if self.Yt is None and self.Zt is None:
            self.cholesky_inverse()
        
        Ybar = np.zeros((self.p,self.p))
        b = 0
        for k in reversed(range(self.n)):
            b = b + self.c[k]**(-2) + self.Zt[:,k]@(Ybar@self.Zt[:,k])
            Ybar = Ybar + np.outer(self.Yt[:,k],self.Yt[:,k])
            
        return b
    
    def trace_inverse_product(self):
        if self.Yt is None and self.Zt is None:
            self.cholesky_inverse()
        b = 0
        P = np.zeros((self.p,self.p))
        R = np.zeros((self.p,self.p))
        
        for k in range(self.n):
            b = b + np.dot(self.Yt[:,k],P@self.Yt[:,k]) + \
                  2*np.dot(self.Yt[:,k],R@self.Ut[:,k])/self.c[k] + \
                    np.dot(self.Ut[:,k],self.Vt[:,k])/(self.c[k]**2)
            P = P + np.dot(self.Ut[:,k],self.Vt[:,k])*np.outer(self.Zt[:,k],self.Zt[:,k]) + \
                    np.outer(self.Zt[:,k],R@self.Ut[:,k]) + \
                    np.outer(R@self.Ut[:,k],self.Zt[:,k])
            R = R + np.outer(self.Zt[:,k],self.Vt[:,k])
        
        return b
