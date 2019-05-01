#!/usr/bin/python

import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt

nx = 41
dx = 2 / (nx-1)
nt = 25
dt = .025
c = 1

# Creates diagonal matrix based on diagonals.

def create_diagonal(left,main,right,size):
   return diags([left, main, right], [-1, 0, 1], shape=(size, size)).toarray()

# Creates diagonal matrix based on diagonals.

def create_vdiag(left,main,right,size):

    A = np.zeros(shape=(size,size))

    np.fill_diagonal(A[1:], left)
    np.fill_diagonal(A[:,:], main)
    np.fill_diagonal(A[:,1:], right)

    return A

# Create initial condition.

u = np.ones(nx)

# Print the initial condition.

print (u)

# Apply initial condition.

u[int(.5 / dx):int(1 / dx + 1)] = 2

# Print initial condition.

print(u)

un = np.ones(nx)

# Iterate the solution and monitor the eigenvalues.

for n in range(nt):

    # To solve numerically, copy the solution.

    un = u.copy()

    # To make the satbility analysis, build the diagonal.
    
    main = np.zeros(nx)

    for i in range(1, nx):
        main[i] = - c * dt / dx * (un[i])

    left = np.zeros(nx)

    for i in range(1, nx):
        left[i] = - c * dt / dx * (- un[i-1])

    right = np.zeros(nx) # No right for this stencil.

    # Build diagonal.

    ss = create_vdiag(left[:],main[:],right[:],nx)

    # Solve the eigenvalues.

    w, v = np.linalg.eig(ss)

    # Prepare the plots.

    imag = np.zeros(nx)
    real = np.zeros(nx)

    real = w.real[:]
    imag = w.imag[:]

    # plot the eigenvalues.

    plt.plot(real, imag, 'ro')
    plt.xlabel('Real(Eig)')
    plt.ylabel('Imag(Eig)')
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()

    # Solve the solution.

    for i in range(1, nx):
        u[i] = un[i] - c * dt / dx * (un[i] - un[i-1])

