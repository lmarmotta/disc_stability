#!/usr/bin/python

# Solves the linear advection equation monitoring the stability properties 
# of the scheme. If it goes unstable, show the solution and the eigenvalues.

import numpy as np
import sys
from scipy.sparse import diags
import matplotlib.pyplot as plt

nx = 41
dx = 2 / (nx-1)
nt = 20
dt = 0.025
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

# Apply initial condition.

u[int(.5 / dx):int(1 / dx + 1)] = 2

# Print initial condition.

un = np.ones(nx)

rhs = np.ones(nx)

frechet_vec = np.ones(nx) 

# Iterate the solution and monitor the eigenvalues.

for n in range(nt):

    # To solve numerically, copy the solution.

    un = u.copy()

    # Separate the residue.

    for i in range(1, nx):
        rhs[i] = ( c * (un[i] - un[i-1])/dx )

    # March the residue.

    for i in range(1, nx):
        u[i] = un[i] - dt*rhs[i]

    # Build the frechet derivative.

    eps = 0.001

    for i in range(1, nx):
        frechet_vec[i] = (( c * (un[i] + eps - un[i-1])/dx ) - ( c * (un[i] - un[i-1])/dx ))/eps

    # Build a diagonal matrix from the derivatives of the residues.

    frechet_matrix = create_vdiag(0,frechet_vec,0,nx)

    # Solve the eigenvalues.

    w, v = np.linalg.eig(frechet_matrix)

    # Prepare the plots.

    imag = np.zeros(nx)
    real = np.zeros(nx)

    real = w.real[:]
    imag = w.imag[:]

    print ("\n")
    print ("Maximun eigenvalues: Real(eig): ", max(real), " Imaginary: Imag(eig): ", max(imag))
    print ("Minimun eigenvalues: Real(eig): ", min(real), " Imaginary: Imag(eig): ", min(imag))

    # plot the eigenvalues.

    fig, ax = plt.subplots(2)
    ax[0].plot(imag, real, 'ro')
    ax[0].set(ylabel='Real(Eig)', xlabel='Imag(Eig)')

    # plot the eigenvalues.

    ax[1].plot(np.linspace(0, 2, nx), u);
    ax[1].set(xlabel='x', ylabel='u')

    plt.show(block=False)
    plt.pause(0.5)
    plt.close()

    if (max(real) > 0.0):
        print ("Code Unstable, Max eigenvalue = ", max(real))
        fig, ax = plt.subplots(2)
        ax[0].plot(imag, real, 'ro')
        ax[0].set(ylabel='Real(Eig)', xlabel='Imag(Eig)')
        ax[1].plot(np.linspace(0, 2, nx), u);
        ax[1].set(xlabel='x', ylabel='u')
        plt.show(block=True)
        sys.exit()

