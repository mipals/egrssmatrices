import unittest
import numpy as np
import numpy.testing as npt
from egrssmatrices.egrssmatrix import egrssmatrix
from egrssmatrices.spline_kernel import spline_kernel

class testegrssmatrices(unittest.TestCase):
    
    def test_matmul(self):
        # Setting up the system
        p,n = 2,50
        t = np.linspace(0.02,1.0,n)
        Ut,Vt = spline_kernel(t,p)
        M  = egrssmatrix(Ut,Vt)
        Md = egrssmatrix(Ut,Vt,t) # Vector diagonal   
        Ms = egrssmatrix(Ut,Vt,1) # Scalar diagonal
        Mfull  = M.full()
        Mdfull = Md.full()
        Msfull = Ms.full()
        # Testing Multiplcation
        x = np.ones(n)
        y  = M@x
        yd = Md@x
        ys = Ms@x
        yfull  = Mfull@x
        ydfull = Mdfull@x
        ysfull = Msfull@x
        npt.assert_almost_equal(y,yfull)
        npt.assert_almost_equal(yd,ydfull)
        npt.assert_almost_equal(ys,ysfull)
        # Testing __getitem__
        npt.assert_equal(Mfull[1,10],  M[1,10])
        npt.assert_equal(Mfull[10,10], M[10,10])
        npt.assert_equal(Mfull[10,1],  M[10,1])
        npt.assert_equal(Mdfull[1,10], Md[1,10])
        npt.assert_equal(Mdfull[10,10],Md[10,10])
        npt.assert_equal(Mdfull[10,1], Md[10,1])
        
    def test_cholesky(self):
        # Setting up the system
        p,n = 2,50
        t = np.linspace(0.02,1.0,n)
        Ut,Vt = spline_kernel(t,p)
        M  = egrssmatrix(Ut,Vt)
        Md = egrssmatrix(Ut,Vt,t)
        Ms = egrssmatrix(Ut,Vt,1)
        Mfull  = M.full()
        Mdfull = Md.full()
        Msfull = Ms.full()
        # Setting up right-hand sides
        x  = np.ones(n)
        y  = M@x
        yd = Md@x
        ys = Ms@x
        # Solving the linear systems
        b  = M.solve(y)
        bd = Md.solve(yd)
        bs = Ms.solve(ys)
        bfull  = np.linalg.solve(Mfull,y)
        bdfull = np.linalg.solve(Mdfull,yd)
        bsfull = np.linalg.solve(Msfull,ys)
        npt.assert_almost_equal(b,bfull)
        npt.assert_almost_equal(bd,bdfull)
        npt.assert_almost_equal(bs,bsfull)
        # Testing traces
        dtracefull = np.matrix.trace(np.linalg.solve(Mdfull,Mfull))
        stracefull = np.matrix.trace(np.linalg.solve(Msfull,Mfull))
        dtrace = Md.trace_inverse_product()
        strace = Ms.trace_inverse_product()
        npt.assert_almost_equal(dtrace,dtracefull)
        npt.assert_almost_equal(strace,stracefull)
        
    def test_cholesky_inverse(self):
        # Setting up the system
        p,n = 2,50
        t = np.linspace(0.02,1.0,n)
        Ut,Vt = spline_kernel(t,p)
        Md = egrssmatrix(Ut,Vt,t)
        Ms = egrssmatrix(Ut,Vt,1)
        Mdfull = Md.full()
        Msfull = Ms.full()
        # Computing the inverse of the Cholesky factor
        Md.cholesky_inverse()
        Ms.cholesky_inverse()
        Tdinv = np.tril(Md.Yt.T@Md.Zt,-1) + np.diag(1/Md.c)
        Tsinv = np.tril(Ms.Yt.T@Ms.Zt,-1) + np.diag(1/Ms.c)
        npt.assert_almost_equal(Tdinv,np.linalg.inv(np.linalg.cholesky(Mdfull)))
        npt.assert_almost_equal(Tsinv,np.linalg.inv(np.linalg.cholesky(Msfull)))
        
    def test_logdet(self):
        # Setting up the system
        p,n = 2,50
        t = np.linspace(0.02,1.0,n)
        Ut,Vt = spline_kernel(t,p)
        M  = egrssmatrix(Ut,Vt)
        Md = egrssmatrix(Ut,Vt,t)
        Ms = egrssmatrix(Ut,Vt,1)
        Mfull  = M.full()
        Mdfull = Md.full()
        Msfull = Ms.full()
        # Computing the log-determinant
        npt.assert_almost_equal(M.logdet(),np.linalg.slogdet(Mfull)[1])
        npt.assert_almost_equal(Md.logdet(),np.linalg.slogdet(Mdfull)[1])
        npt.assert_almost_equal(Ms.logdet(),np.linalg.slogdet(Msfull)[1])
        
if __name__ == '__main__':
    unittest.main()
