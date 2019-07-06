#!/usr/bin/python

# Solves the diffusion equation using a simple centered scheme.
# While the iterations proceed, compute the eigenvalues of the 
# D_RHS/D_U Jacobian in order to understand the spatial discre
# tization stability.

import autograd.numpy as np
from autograd import jacobian
import sys
import os
from scipy.sparse import diags
import matplotlib.pyplot as plt
from autograd import grad, jacobian

nx = 15
dx = 2 / (nx-1)
nt = 150
nu = 0.1
dt = 0.01

# Creates diagonal matrix based on diagonals.

def create_diagonal(left,main,right,size):
   return diags([left, main, right], [-1, 0, 1], shape=(size, size)).toarray()

# Residue function of a simple centered three point scheme.

def frhs(um1,uo1,up1,dx,nu):
    return nu*( (up1 - 2.0 * uo1 + um1)/dx**2.0 )

# Verctorial version of the residue.

def frhs_vec(u,nu,dx):

    u_out = np.zeros(len(u))

    for i in range(len(u)-1):
        u_out[i] = nu*( ( u[i+1] - 2.0 * u[i] + u[i-1] ) /dx**2.0)

    return u_out

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

        # Dump iteration count.

        print (" +++ Time: " + str(n) + " +++")

        # Copy the solution of the explicit time marching scheme.

        un = u.copy()

        # Separate the residue.

        rhs = frhs_vec(un,nu,dx)

        # March the residue.

        u = un + dt*rhs

        # Computes the derivative of the residues with respect to the solution vector.

        eps = 0.0001

        drhs_du = np.zeros((nx-1,nx-1))  # In order to take the eigenvalues, this shall be a matrix.

        # This loop computes the jacobian matrix according to http://www.netlib.org/math/docpdf/ch08-04.pdf

        for i in range(1,nx-1):
            for j in range(1,nx-1):
                drhs_du[i,j] = ( frhs(un[i-1]+eps, un[i]+eps, un[i+1]+eps,dx,nu) - frhs(un[j-1], un[j],un[j+1],dx,nu) )/eps

        # Build the Hirsch matrix (chap 8).

        s_m = np.zeros((nx-1,nx-1))

        # Fill the diagonals

        s_m = (nu/dx**2.0)*create_diagonal(1.0,-2.0,1.0,nx-1)


        # Solve the eigenvalues.

        w1, v1 = np.linalg.eig(drhs_du)
        w2, v2 = np.linalg.eig(s_m)

        # Prepare the plots.

        real1 = np.zeros(nx)
        imag1 = np.zeros(nx)

        real1 = -np.sort(-w1.real[:])
        imag1 = -np.sort(-w1.imag[:])

        real2 = np.zeros(nx)
        imag2 = np.zeros(nx)

        real2 = -np.sort(-w2.real[:])
        imag2 = -np.sort(-w2.imag[:])

        print ("\n")
        print ("Minimun eigenvalues (Frechet): Real(eig): ", min(real1), " Imaginary: Imag(eig): ", min(imag1))
        print ("Maximun eigenvalues (Frechet): Real(eig): ", max(real1), " Imaginary: Imag(eig): ", max(imag1))
        print ("Minimun eigenvalues (Hirsch ): Real(eig): ", min(real2), " Imaginary: Imag(eig): ", min(imag2))
        print ("Maximun eigenvalues (Hirsch ): Real(eig): ", max(real2), " Imaginary: Imag(eig): ", max(imag2))

    # Print both matrices.

    print (np.matrix(s_m))
    print ("------------------------------------------------------------")
    print (np.matrix(drhs_du))

    # plot the eigenvalues.

    plt.figure(3)
    fig, ax = plt.subplots(3,figsize=(11, 11))
    ax[0].plot(imag1, real1, 'ro')
    ax[0].set(ylabel='Real(Eig)', xlabel='Imag(Eig)')
    ax[0].set_xlim(-0.06,0.06)
    # ax[0].set_ylim(-70.0,10.0)

    ax[1].plot(imag2, real2, 'ro')
    ax[1].set(ylabel='Real(Eig)', xlabel='Imag(Eig)')
    ax[1].set_xlim(-0.06,0.06)
    # ax[1].set_ylim(-70.0,10.0)

    ax[2].plot(np.linspace(0, 2, nx), u);
    ax[2].set(xlabel='x', ylabel='u')

    image_name = str(n) + "image"  + ".png"

    plt.savefig(image_name)
    plt.close()

if __name__ == '__main__':
    main()
