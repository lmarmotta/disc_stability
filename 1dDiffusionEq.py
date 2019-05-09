#!/usr/bin/python

# Solves the diffusion equation using a simple centered scheme.
# While the iterations proceed, compute the eigenvalues of the 
# D_RHS/D_U Jacobian in order to understand the spatial discre
# tization stability.

import numpy as np
import sys
from scipy.sparse import diags
import matplotlib.pyplot as plt
from autograd import grad, jacobian

# Stable setup.

nx = 41
dx = 2 / (nx-1)
nt = 30
nu = 0.1
dt = 0.001

# Creates diagonal matrix based on diagonals.

def create_diagonal(left,main,right,size):
   return diags([left, main, right], [-1, 0, 1], shape=(size, size)).toarray()

# Residue function of a simple centered three point scheme.

def frhs(um1,uo1,up1,dx,nu):
    return nu*( (up1 - 2.0 * uo1 + um1)/dx**2.0 )

# Main function.

def main():

    # Create initial condition.

    u = np.ones(nx)

    # Apply the step condition.

    u[int(.5 / dx):int(1 / dx + 1)] = 2

    # Print initial condition.

    un = np.ones(nx)

    # Prepare the residue vector.

    rhs = np.zeros(nx)

    # Iterate the solution and monitor the eigenvalues.

    for n in range(nt):

        # Copy the solution of the explicit time marching scheme.

        un = u.copy()

        # Separate the residue.

        for i in range(1, nx-1):
            rhs[i] = frhs(un[i-1],un[i],un[i+1],dx,nu)

        # March the residue.

        for i in range(1, nx-1):
            u[i] = un[i] + dt*rhs[i]

        # Computes the derivative of the residues with respect to the solution vector.

        eps = 0.001

        drhs_du = np.zeros((nx-1,nx-1))  # In order to take the eigenvalues, this shall be a matrix.

        for i in range(1,nx-1):
            for j in range(1,nx-1):
                drhs_du[i,j] = 0.00001
                drhs_du[i,j] = ( un[i]*frhs(un[i-1] + eps,un[i] + eps,un[i+1] + eps,dx,nu) - un[j]*frhs(un[j-1],un[j],un[j+1],dx,nu) )/eps

        # Build the Hirsch matrix.

        s_m = np.zeros((nx-1,nx-1))

        # Fill the diagonals

        s_m = (nu/dx**2.0)*create_diagonal(1.0,-2.0,1.0,nx-1)

        # Print both matrices.

        print (np.matrix(s_m))
        print ("------------------------------------------------------------")
        print (np.matrix(drhs_du))

        # Solve the eigenvalues.

        w1, v1 = np.linalg.eig(drhs_du)
        w2, v2 = np.linalg.eig(s_m)

        # Prepare the plots.

        real1 = np.zeros(nx)
        imag1 = np.zeros(nx)

        real1 = w1.real[:]
        imag1 = w1.imag[:]

        real2 = np.zeros(nx)
        imag2 = np.zeros(nx)

        real2 = w2.real[:]
        imag2 = w2.imag[:]

        print ("\n")
        print ("Minimun eigenvalues (Frechet): Real(eig): ", min(real1), " Imaginary: Imag(eig): ", min(imag1))
        print ("Maximun eigenvalues (Frechet): Real(eig): ", max(real1), " Imaginary: Imag(eig): ", max(imag1))
        print ("Minimun eigenvalues (Hirsch ): Real(eig): ", min(real2), " Imaginary: Imag(eig): ", min(imag2))
        print ("Maximun eigenvalues (Hirsch ): Real(eig): ", max(real2), " Imaginary: Imag(eig): ", max(imag2))

        # plot the eigenvalues.

        plt.figure(3)
        fig, ax = plt.subplots(3,figsize=(20, 20))
        ax[0].plot(imag1, real1, 'ro')
        ax[0].set(ylabel='Real(Eig)', xlabel='Imag(Eig)')

        ax[1].plot(imag2, real2, 'ro')
        ax[1].set(ylabel='Real(Eig)', xlabel='Imag(Eig)')

        ax[2].plot(np.linspace(0, 2, nx), u);
        ax[2].set(xlabel='x', ylabel='u')

        plt.show(block=False)
        plt.pause(0.5)

        image_name = "image_" + str(n) + ".png"

        plt.savefig(image_name)
        plt.close()

if __name__ == '__main__':
    main()
