import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from scipy import optimize
from egrssmatrices.spline_kernel import spline_kernel
from egrssmatrices.egrssmatrix import egrssmatrix

# Evaluating polynomial basis
def evaluate_polynomial_basis(t,p):
    n = len(t)
    H = np.ones((n,p),t.dtype)
    for i in range(2,p+1):
        H[:,i-1] = t**(i - 1)/factorial(i-1)
    return H

# Computing coeffieints and log-generalized-maximum-likelihood
def compute_coefficients(K, H, y, alpha):
    n, p = H.shape
    chol = egrssmatrix(K.Ut,K.Vt,n*alpha)
    chol.cholesky()
    KinvH = np.zeros((n,p))
    # For now solve only works with vectors, so we have to loop over the columns of H
    for k in range(p):
        KinvH[:,k] = chol.solve(H[:,k])
    A = H.T@KinvH
    v = chol.solve(y)
    d = np.linalg.solve(A,H.T@v) # Dense computation, but A is p x p, so its not a problem
    c = chol.solve(y - H@d)      # Efficient solve using the O(p^2n) algorithm.
    log_gml = np.log(np.dot(y,c)) + chol.logdet()/(n - p) + np.linalg.slogdet(A)[1]/(n - p)
    return c,d,log_gml

# In the following we will fit the Forrester et al. (2008) Function
f = lambda t: (6*t - 2)**2 * np.sin(12*t - 4)

# Defining data
n       = 100                            # Number of data points
sigma   = 2.0                            # Noise standard deviation
t = np.sort(np.random.rand(n))           # Generating random x-data in the interval
y = f(t) + sigma*np.random.randn(len(t)) # Adding noise to the y-data

# Defining the exact spline
p = 2                               # Defining spline order
Ut,Vt = spline_kernel(t,p)          # Computing the generators
K = egrssmatrix(Ut,Vt)              # Defining the EGRSS matrix
H = evaluate_polynomial_basis(t,p)  # Computing polynomial basis

# Log generalized maximum likelihood
def log_gml(v):
    _,_,log_gml = compute_coefficients(K,H,y,10**(v))
    return log_gml

# Minimizing log-gml 
minimizer = optimize.minimize_scalar(log_gml,method="brent")
# Recomputing coefficients
alpha = 10**(minimizer.x)
c,d,_ = compute_coefficients(K,H,y,alpha)
# Computing fit
smoothed_fit = K@c + H@d

plt.plot(t,f(t), label="True function")
plt.plot(t,smoothed_fit, label="Spline fit")
plt.scatter(t,y, label = "Data")
plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.show()