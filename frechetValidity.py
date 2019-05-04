#!/usr/bin/python

# Solves the linear advection equation monitoring the stability properties 
# of the scheme. If it goes unstable, show the solution and the eigenvalues.

import numpy as np
import sys
from scipy.sparse import diags
import matplotlib.pyplot as plt

# Stable setup.

nx = 41
dx = 2 / (nx-1)
nt = 25
dt = 0.025
c = 1

# Unstable setup.
# 
# nx = 41
# dx = 2 / (nx-1)
# nt = 25
# dt = 0.060
# c = 1

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

def frhs(unn,unm1,dx):
    return (unn - unm1)/dx

# Create initial condition.

u = np.ones(nx)

# Apply initial condition.

u[int(.5 / dx):int(1 / dx + 1)] = 2

# Print initial condition.

un = np.ones(nx)

rhs = np.ones(nx)

# Iterate the solution and monitor the eigenvalues.

for n in range(nt):

    # To solve numerically, copy the solution.

    un = u.copy()

    # Separate the residue.

    for i in range(1, nx-1):
        rhs[i] = c * frhs(un[i],un[i-1],dx)  # Backward, stable.
        # rhs[i] = c * frhs(un[i+1],un[i-1],2.0*dx)  # Central, unstable.

    # March the residue.

    for i in range(1, nx-1):
        u[i] = un[i] - dt*rhs[i]

    # Computes the derivative of the residues with respect to the solution vector.

    eps = 0.0001

    drhs_du = np.zeros((nx,nx))  # In order to take the eigenvalues, this shall be a matrix.

    for i in range(1,nx-1):
        for j in range(1,nx-1):
            drhs_du[i,j] = (frhs(un[i] + eps,un[i-1] + eps,dx) - frhs(un[j],un[j-1],dx))/eps

    # Solve the eigenvalues.

    w1, v1 = np.linalg.eig(drhs_du)

    # Prepare the plots.

    real1 = np.zeros(nx)
    imag1 = np.zeros(nx)

    real1 = w1.real[:]
    imag1 = w1.imag[:]

    print ("\n")
    print ("Maximun eigenvalues: Real(eig): ", max(real1), " Imaginary: Imag(eig): ", max(imag1))

    # plot the eigenvalues.

    fig, ax = plt.subplots(2)
    ax[0].plot(imag1, real1, 'ro')
    ax[0].set(ylabel='Real(Eig)', xlabel='Imag(Eig)')

    # plot the eigenvalues.

    ax[1].plot(np.linspace(0, 2, nx), u);
    ax[1].set(xlabel='x', ylabel='u')

    plt.show(block=False)
    plt.pause(0.5)
    plt.close()
