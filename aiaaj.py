#!/usr/bin/python

# Recreating Pullians paper on stability of artificial dissipation 
# schemes.
#
# VOL. 24, NO. 12, DECEMBER 1986
# Artificial Dissipation Models for the Euler Equations

import numpy as np
import sys
from scipy.sparse import diags
import matplotlib.pyplot as plt

size_p = 8

# Creates diagonal matrix based on diagonals.

def create_diagonal(Left,left,main,right,Right,size):
   return diags([Left, left, main, right, Right], [-2,-1, 0, 1, 2], shape=(size, size)).toarray()

A = create_diagonal(-1,4,-6,4,-1,size_p)

print (A)

# Solve the eigenvalues.

w, v = np.linalg.eig(A)

# Prepare the plots.

imag = np.zeros(size_p)
real = np.zeros(size_p)

imag = w.imag[:]
real = w.real[:]

# Matrix 43a.

for i in range(size_p):
    print ("IM(eig): ", imag[i], "RE(eig): ", real[i])


