import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import cm
import os 
import imageio.v2 as imageio
import ansatz
import potentialFuncs

def Psi(x,y,tau=0):
    r=np.sqrt(x**2+y**2)
    return r*np.exp(-r**2)

def potential(x,y,m,omega0):
    return 1/2 * m * omega0**2 * (x**2+y**2)

def nonLin(psiVals, potentialVals, gamma, m, dtau):
    return gamma*np.abs(psiVals)**2+potentialVals

def nextStep(psiVals, potentialFunc:callable, gamma, m, omega0, L, dL ,dtau,  tauMax, XX, YY, kXX, kYY):
    """
    Implement Strang Splitting. Take one step in imaginary time towards ground state
    """

    potentialVals = potentialFunc(XX, YY, m, omega0)

    nonlinVals=nonLin(psiVals, potentialVals, gamma, m, dtau)
    nonLin_expPsi=np.exp(-nonlinVals*dtau/2)*psiVals
    FT_nonLin_expPsi = np.fft.fft2(nonLin_expPsi)

    FT_dispersion = (kXX**2+kYY**2)/ (2*m)
    exp_FT_dispersion = np.exp(-dtau * FT_dispersion)
    newPsi = exp_FT_dispersion*FT_nonLin_expPsi

    posSpace_Psi = np.fft.ifft2(newPsi)
    nonLin_newVals = nonLin(posSpace_Psi, potentialVals, gamma, m, dtau)
    exp_nonLin_newVals = np.exp(-dtau/2 * nonLin_newVals)

    finalPsi = exp_nonLin_newVals * posSpace_Psi

    normalisedPsi = finalPsi/np.sqrt(np.sum(np.abs(finalPsi)**2)* dL**2)

    return normalisedPsi.astype(np.complex64)

def loopy(psiVals, potentialFunc:callable, gamma, m, omega0, L, dL ,dtau,  tauMax:int, XX, YY, kXX, kYY, solve:bool=False):
    """
    Compute nextstep many times until  tauMax is reached to estimate a ground state
    """
    steps=[psiVals]
    taus = [0]
    curTau=0
    curStep = psiVals
    while curTau <=  tauMax:
        if solve:
            print(30*"\n", 50*"=")
            print(" Bose-Einstein Condensate Dynamics Simulator")
            print(" STEP 0 of 4 - Finding Ground State")
            print("", 50*"=")
        else:
            print(30*"\n", "Finding ground state...")
        print(" Propagating Imaginary time...  \n Iteration", int(curTau/dtau), " out of ", int(tauMax/dtau))
        curStep = nextStep(curStep, potentialFunc, gamma, m, omega0, L, dL ,dtau,  tauMax, XX, YY, kXX, kYY)
        steps.append(curStep)
        taus.append(curTau)
        if np.isnan(curStep).any():
            print("The wavefunction has diverged!")
            return steps[-2], taus[:-1], steps[:-1], True
        curTau+=dtau
    print(30*"\n", "Finding ground state in imaginary time...")
    print(" Advancing in Imaginary time.  Step", int(100*curTau/tauMax), " of ", int(tauMax/dtau))
    return curStep, taus, steps, False

def plotter(groundVals, dL, L, plottingBorders):
    print(30*"\n")
    print("Ground state found. Plotting results...")
    
    X=np.linspace(-L, L, int(2*L/dL+1))
    Y=np.linspace(-L, L, int(2*L/dL+1))

    interiorXIndices = np.where(np.abs(X) <= plottingBorders)[0]
    interiorYIndices = np.where(np.abs(Y) <= plottingBorders)[0]
    interiorX=X[interiorXIndices]
    interiorY=Y[interiorYIndices]
    interiorXX, interiorYY = np.meshgrid(interiorX, interiorY)

    psiVals=np.array(groundVals)
    densities = np.abs(psiVals)**2
    interiorDensities = densities[np.ix_(interiorYIndices, interiorXIndices)]
    maxDensity = np.max(interiorDensities)
    minDensity = max(1e-6, np.min(interiorDensities))


    plt.figure(figsize=(8,5))
    img = plt.pcolor(interiorXX, interiorYY, interiorDensities, cmap="nipy_spectral", norm=LogNorm(vmin= minDensity, vmax=maxDensity))
    plt.title(f"Imaginary Time Ground State for Bose-Einstein Condensate")
    plt.xlabel("x-position")
    plt.ylabel("y-position")
    plt.colorbar(img, label="Condensate Density")
    print("Plotting complete. See output graph for results.")
    plt.show()

def groundFinder(ansatzFunc:callable, potentialFunc:callable, gamma, m, omega0, L, dL ,dtau,  tauMax, plottingBorders, plotOrNot:bool):
    print("Initialising ground state finder...")
    X=np.linspace(-L, L, int(2*L/dL+1))
    Y=np.linspace(-L, L, int(2*L/dL+1))
    XX, YY = np.meshgrid(X, Y)

    kX=np.fft.fftfreq(len(X), d=dL)
    kY=np.fft.fftfreq(len(Y), d=dL)
    kXX, kYY = np.meshgrid(kX, kY)

    ansatzVals = ansatzFunc(XX, YY)

    groundVals, taus, steps, diverge = loopy(ansatzVals, potentialFunc, gamma, m, omega0, L, dL ,dtau,  tauMax, XX, YY, kXX, kYY, not plotOrNot)

    if plotOrNot:
        plotter(groundVals, dL, L, plottingBorders)
    return groundVals, diverge

#groundFinder(ansatz.vortexGaussian, potentialFuncs.harmonicOscillator, -1, 1, 1/3, 5, 0.01, 0.01, 10 , 3/2, True)

#MINIMUM: tauMax ~ 1/omega0, for safety and accuracy reccommend tauMax ~ 2 * 1/omega0