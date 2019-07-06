#!/usr/bin/python

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

nx = 250
dx = 2 / (nx - 1)
nt = 300
dt = 0.001
c  = 1.0

# Residue function of a simple centered three point scheme.

def frhs(uo1,um1,dx):
    return c*(uo1 - um1)/dx

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

        for i in range(1,nx-1):
            rhs[i] = frhs(un[i],un[i-1],dx)

        # March the residue.

        for i in range(1,nx-1):
            u[i] = un[i] - dt*rhs[i]

        # Computes the derivative of the residues with respect to the solution vector.

        eps = 0.0001

        drhs_du = np.zeros((nx-1,nx-1))  # In order to take the eigenvalues, this shall be a matrix.

        for i in range(1,nx-1):
            for j in range(1,nx-1):
                drhs_du[i,j] = ( frhs(un[i]+eps, un[i-1]+eps,dx) - frhs(un[j]-eps, un[j-1]-eps,dx) )/2.0*eps

        # Solve the eigenvalues.

        w1, v1 = np.linalg.eig(drhs_du)

        # Prepare the plots.

        real1 = w1.real[:]
        imag1 = w1.imag[:]

        print ("\n")
        print ("Minimun eigenvalues (Frechet): Real(eig): ", min(real1), " Imaginary: Imag(eig): ", min(imag1))
        print ("Maximun eigenvalues (Frechet): Real(eig): ", max(real1), " Imaginary: Imag(eig): ", max(imag1))

        # If we have a bigger than reasonable eigenvalues, than print what we got and get out !

        if (max(real1) > 5.0):

            plt.figure(2)
            fig, ax = plt.subplots(2,figsize=(11, 11))
            ax[0].plot(imag1, real1, 'ro')
            ax[0].set(ylabel='Real(Eig)', xlabel='Imag(Eig)')
            ax[0].set_xlim(-0.06,0.06)

            ax[1].plot(np.linspace(0, 2, nx), u);
            ax[1].set(xlabel='x', ylabel='u')

            image_name = str(n) + "image"  + ".png"

            plt.savefig(image_name)
            plt.close()

            break

    # plot the eigenvalues.

    plt.figure(2)
    fig, ax = plt.subplots(2,figsize=(11, 11))
    ax[0].plot(imag1, real1, 'ro')
    ax[0].set(ylabel='Real(Eig)', xlabel='Imag(Eig)')
    ax[0].set_xlim(-0.06,0.06)

    ax[1].plot(np.linspace(0, 2, nx), u);
    ax[1].set(xlabel='x', ylabel='u')

    image_name = str(n) + "image"  + ".png"

    plt.savefig(image_name)
    plt.close()

if __name__ == '__main__':
    main()
