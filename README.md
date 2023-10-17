# egrssmatrices

[![Build Status](https://github.com/mipals/egrssmatrices/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/mipals/egrssmatrices/actions/workflows/python-app.yml?query=branch%3Amain)

## Description
A package for efficiently computing with symmetric extended generator representable semiseparable matrices (EGRSS Matrices) and a variant with added diagonal terms. In short this means matrices of the form

$$
K = \text{\textbf{tril}}(UV^\top) + \text{\textbf{triu}}\left(VU^\top,1\right), \quad U,V\in\mathbb{R}^{n\times p}
$$

$$
K = \text{\textbf{tril}}(UV^\top) + \text{\textbf{triu}}\left(VU^\top,1\right) + \text{\textbf{diag}}(d), \quad U,V\in\mathbb{R}^{n\times p},\ d\in\mathbb{R}^n
$$

All implemented algorithms (multiplication, Cholesky factorization, forward/backward substitution as well as various traces and determinants) scales with $O(p^kn)$. Since $p \ll n$ this result in very scalable computations.

A more in-depth descriptions of the algorithms can be found in [1] or [here](https://github.com/mipals/SmoothingSplines.jl/blob/master/mt_mikkel_paltorp.pdf).

## References
[1] M. S. Andersen and T. Chen, “Smoothing Splines and Rank Structured Matrices: Revisiting the Spline Kernel,” SIAM Journal on Matrix Analysis and Applications, 2020.
