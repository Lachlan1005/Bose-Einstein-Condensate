import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import cm
import os 
import imageio.v2 as imageio
import ansatz
import potentialFuncs
import initstateFinder
import perturbations

def Psi(x:float, y:float, t:float=0)->float:
    """
    Return the value of the intial wavefunction at position (x,y) and time t
    """
    #return np.exp(-3*((x-2/3)**2+(y-2/3)**2))+np.exp(-3*((x+2/3)**2+(y+2/3)**2))
    return np.exp(1j*2*np.arctan2(y, x))*np.sqrt(x**2+y**2)*np.exp(-5*((x-0)**2+y**2))

def potential(x:float, y:float, m:float, omega0:float)->float:
    """
    Return the potential at position (x,y) for a particle with mass m and natural angular frequency omega0
    """
    return 1/2 *m*omega0**2 * (x**2+y**2) 

def vortexPerturbation(x,y):
    l=1
    theta=np.arctan2(y,x)
    return np.exp(1j*l*theta)

def nonLinear(psiVal:float, potentialVal:float, gamma:float, m:float, omega0:float)->float:
    """
    Return tthe nonLinear operator of the Gross-Pitaeskvskii equation
    """
    return gamma*abs(psiVal)**2 + potentialVal

def nextStep(psiVals:np.ndarray, L:float,  dt:float, dL:float,  gamma:float, m:float, omega0:float, customPotential:callable=potential)->np.ndarray:
    """
    Treat the nonLinear and dispersion components with the split step Fourier method to obtain the next value for the wavefunction within a square area LxL
    Note that dx=dy=dL 
    Note that to implement things in 2D need to use meshgrid and to Fourier transform in 2D need fft2
    (meshgrid X -> assign X coordinate for every point in space )
    (meshgrid Y -> assign Y coordinate for every point in space )
    """
    X=np.linspace(-L,L,int(2*L/dL +1))
    Y=np.linspace(-L,L, int(2*L/dL +1))
    XX, YY = np.meshgrid(X, Y)
    potentials=customPotential(XX,YY, m, omega0)

    kX=2*np.pi*np.fft.fftfreq(len(X), d=dL)
    kY=2*np.pi*np.fft.fftfreq(len(Y), d=dL)
    kXX, kYY= np.meshgrid(kX, kY)

    nonLinearVals=nonLinear(psiVals, potentials, gamma, m, omega0)
    nonLin_expVals=np.exp(-1j*dt*nonLinearVals/2)

    nonLin_Psi=nonLin_expVals*psiVals

    firstStep_nonLin_Fourier=np.fft.fft2(nonLin_Psi)
    dispersion_Fourier = np.exp(-1j*dt/(2*m) * (kXX**2+kYY**2))

    psiStep_posSpace=np.fft.ifft2(dispersion_Fourier*firstStep_nonLin_Fourier)

    nonLinear_Vals2= nonLinear(psiStep_posSpace, potentials, gamma, m, omega0)
    nonLin_expVals2 = np.exp(-1j * dt * nonLinear_Vals2 / 2)

    return nonLin_expVals2*psiStep_posSpace

def loopy(tmax:float, initialPsi:callable, L:float,  dt:float, dL:float,  gamma:float, m:float, omega0:float, customPotential:callable=potential):
    print("Initialising Solver...")
    X=np.linspace(-L,L,int(2*L/dL +1))
    Y=np.linspace(-L,L, int(2*L/dL +1))
    t=0
    tList=[0]
    XX, YY = np.meshgrid(X, Y)
    psiVals=initialPsi(XX,YY).astype(np.complex64)
    psiVals=psiVals/np.sqrt(np.sum(np.abs(psiVals)**2)* dL**2   )
    wavefuncs=[psiVals]
    while t<=tmax:
        print(30*"\n","Solver running...")
        print(f" Current Simulation time: {t:.3f} || Maximum simulation time: {tmax:.3f} || Progress: {(100 * t / tmax):.2f}%")
        psiVals=nextStep(psiVals, L, dt, dL, gamma, m, omega0,customPotential)
        psiVals=psiVals/np.sqrt(np.sum(np.abs(psiVals)**2)* dL**2)
        t+=dt
        wavefuncs.append(psiVals)
        tList.append(t)
    print(30*"\n")
    print(f" Current Simulation time: {tmax:.3f} || Maximum simulation time: {tmax:.3f} || Progress: {(100):.2f}%")
    print(" Solving complete. Proceed to post-process if visualisation is necessary.")
    return tList, X, Y, wavefuncs

def plotter(tmax:float, initialPsi:callable, L:float,  dt:float, dL:float,  gamma:float, m:float, customParam:float, plottingBorders:float, fpsCustom:float=100, customPotential:callable=potential):
    """
    Plot the outputs from loopy and create a video representation of the time evolution. Plot in 3D within a plottingBorders x plottingBorders grid
    customParam denotes a free parameter custom to the potential. For example, in the Harmonic potential, 
    customParam denotes the natural frequency. In the square well, it denotes the length of the well, etc. 
    See more in the potentialFuncs file. 
    """
    tList, X, Y, wavefuncs = loopy(tmax, initialPsi, L, dt, dL, gamma, m, customParam, customPotential)

    print(" Post-Processing solution...")
    interiorXIndices = np.where(np.abs(X) <= plottingBorders)[0]
    interiorYIndices = np.where(np.abs(Y) <= plottingBorders)[0]
    interiorX=X[interiorXIndices]
    interiorY=Y[interiorYIndices]
    interiorXX, interiorYY = np.meshgrid(interiorX, interiorY)

    wavefuncs=np.array(wavefuncs)
    densities = np.abs(wavefuncs)**2
    interiorDensities = densities[:, interiorYIndices[:, None], interiorXIndices]

    os.makedirs("frames", exist_ok=True)

    i=0 
    maxDensity = np.max(interiorDensities)
    while i<len(tList):
        print(30*"\n","Post-Processing complete. Plotting results and constructing frames...")
        print(f" {i} out of {len(tList)} frames constructed ({(100 * i / len(tList)):.2f}% )")
        curDensities = interiorDensities[i]
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(interiorXX, interiorYY, curDensities,cmap=cm.nipy_spectral)
        plt.title(f"2D Solution for Gross-Pitaevskii Model of Bose-Einstein Condensate, t={tList[i]:.3f} (Frame {i})")
        ax.set_xlabel("X-Position")
        ax.set_ylabel("Y-Position")
        ax.set_zlabel("Condensate Density")
        ax.set_zlim(0, maxDensity) 
        frame_path = f"frames/frame_{i:04d}.png"
        plt.savefig(frame_path)
        plt.close()
        i+=1 
    print(30*"\n")
    print(f" {len(tList)} out of {len(tList)} frames constructed ({100:.2f}% )")
    print(" Frames Constructed. Initialising next process...")

    j=0 
    output_path ="/Users/kanlachlan/Documents/VS_Code/Personal Projects/Bose-Einstein_Condensates/2D_Sim_Output.mp4"
    with imageio.get_writer(output_path, fps=fpsCustom) as writer:
        while j<len(tList):
            print(30*"\n", "Frames constructed. Assembling video...")
            print(f" {j} out of {len(tList)} frames assembled ({(100 * j / len(tList)):.2f}% )")
            frame_path = f"frames/frame_{j:04d}.png"
            image = imageio.imread(frame_path)
            writer.append_data(image)
            j+=1
    print(30*"\n", "Frames constructed. Assembling video...")
    print(f" {len(tList)} out of {len(tList)} frames assembled ({100.00:.2f}% )")
    print(" Video saved as 2D_Sim_Output.mp4 with path ", output_path)


def plotterFLAT(tmax:float, initialPsi:callable, L:float,  dt:float, dL:float,  gamma:float, m:float, customParam:float, plottingBorders:float, fpsCustom:float=100, customPotential:callable=potential):
    """
    Plot (in a 2D colour map) the outputs from loopy and create a video representation of the time evolution. Plot in 3D within a plottingBorders x plottingBorders grid
    customParam denotes a free parameter custom to the potential. For example, in the Harmonic potential, 
    customParam denotes the natural frequency. In the square well, it denotes the length of the well, etc. 
    See more in the potentialFuncs file. 
    """
    tList, X, Y, wavefuncs = loopy(tmax, initialPsi, L, dt, dL, gamma, m, customParam, customPotential)

    print(" Post-Processing solution...")
    interiorXIndices = np.where(np.abs(X) <= plottingBorders)[0]
    interiorYIndices = np.where(np.abs(Y) <= plottingBorders)[0]
    interiorX=X[interiorXIndices]
    interiorY=Y[interiorYIndices]
    interiorXX, interiorYY = np.meshgrid(interiorX, interiorY)

    wavefuncs=np.array(wavefuncs)
    densities = np.abs(wavefuncs)**2
    interiorDensities = densities[:, interiorYIndices[:, None], interiorXIndices]

    os.makedirs("frames", exist_ok=True)

    i=0 
    maxDensity = np.max(interiorDensities)
    while i<len(tList):
        print(30*"\n","Post-Processing complete. Plotting results and constructing frames...")
        print(f" {i} out of {len(tList)} frames constructed ({(100 * i / len(tList)):.2f}% )")
        curDensities = interiorDensities[i]
        plt.figure(figsize=(8,5))
        img = plt.pcolor(interiorXX, interiorYY, curDensities, cmap="nipy_spectral", norm=LogNorm(vmin=1e-4, vmax=maxDensity))
        #img = plt.pcolor(interiorXX, interiorYY, curDensities, cmap="nipy_spectral", vmin=0, vmax=maxDensity)
        plt.title(f"2D Solution for GPE Model of Bose-Einstein Condensate, t={tList[i]:.3f} (Frame {i})")
        #plt.title(f"2D Solution for GPE Model of Bose-Einstein Condensate")
        plt.xlabel("x-position")
        plt.ylabel("y-position")
        plt.colorbar(img, label="Condensate Density")
        frame_path = f"frames/frame_{i:04d}.png"
        plt.savefig(frame_path)
        plt.close()
        i+=1 
    print(30*"\n")
    print(f" {len(tList)} out of {len(tList)} frames constructed ({100:.2f}% )")
    print(" Frames Constructed. Initialising next process...")

    j=0 
    output_path ="/Users/kanlachlan/Documents/VS_Code/Personal Projects/Bose-Einstein_Condensates/2D_FLAT_Sim_Output.mp4"
    with imageio.get_writer(output_path, fps=fpsCustom) as writer:
        while j<len(tList):
            print(30*"\n", "Frames constructed. Assembling video...")
            print(f" {j} out of {len(tList)} frames assembled ({(100 * j / len(tList)):.2f}% )")
            frame_path = f"frames/frame_{j:04d}.png"
            image = imageio.imread(frame_path)
            writer.append_data(image)
            j+=1
    print(30*"\n", "Frames constructed. Assembling video...")
    print(f" {len(tList)} out of {len(tList)} frames assembled ({100.00:.2f}% )")
    print(" Video saved as 2D_Sim_Output.mp4 with path ", output_path)



def loopy2(tmax:float, initialPsi:np.ndarray, L:float,  dt:float, dL:float,  gamma:float, m:float, omega0:float, customPotential:callable=potential, diverge:bool=False, perterbation:callable=vortexPerturbation):
    """
    This serves the same function as loopy2 except initialPsi must be an ARRAY, not a callable
    """
    print("Initialising Solver...")
    X=np.linspace(-L,L,int(2*L/dL +1))
    Y=np.linspace(-L,L, int(2*L/dL +1))
    XX, YY = np.meshgrid(X, Y)
    t=0
    tList=[0]

    perturbationVals = perterbation(XX, YY)
    psiVals=initialPsi*perturbationVals
    wavefuncs=[psiVals]

    while t<=tmax:
        print(30*"\n", 50*"=")
        print(" Bose-Einstein Condensate Dynamics Simulator")
        print(" STEP 2 of 4 - Simulating Perturbed Ground State Dynamics")
        print("", 50*"=")
        if diverge:
            print(" WARNING: Working with a DIVERGED initial condition!") 
        print(" Propagating real time...")
        print(f" Current Simulation time: {t:.3f} || Maximum simulation time: {tmax:.3f} || Progress: {(100 * t / tmax):.2f}%")
        psiVals=nextStep(psiVals, L, dt, dL, gamma, m, omega0,customPotential)
        psiVals=psiVals/np.sqrt(np.sum(np.abs(psiVals)**2)* dL**2)
        t+=dt
        wavefuncs.append(psiVals)
        tList.append(t)
    print(30*"\n", 50*"=")
    print(" Bose-Einstein Condensate Dynamics Simulator")
    print(" STEP 2 of 4 - Simulating Dynamics")
    print("", 50*"=")
    print(f" Current Simulation time: {tmax:.3f} || Maximum simulation time: {tmax:.3f} || Progress: {(100):.2f}%")
    print(" Solving complete. Proceed to post-process if visualisation is necessary.")
    return tList, X, Y, wavefuncs


def plotterFLAT2(tmax:float, initialPsi:np.ndarray, L:float,  dt:float, dL:float,  gamma:float, m:float, customParam:float, plottingBorders:float, fpsCustom:float=100, customPotential:callable=potential, diverge:bool=False, perterbation:callable=vortexPerturbation):
    """
    Same as PlotterFLAT, except again initialPsi is an ARRAY, not a callable
    """
    tList, X, Y, wavefuncs = loopy2(tmax, initialPsi, L, dt, dL, gamma, m, customParam, customPotential, diverge, perterbation)

    print(" Post-Processing solution...")
    interiorXIndices = np.where(np.abs(X) <= plottingBorders)[0]
    interiorYIndices = np.where(np.abs(Y) <= plottingBorders)[0]
    interiorX=X[interiorXIndices]
    interiorY=Y[interiorYIndices]
    interiorXX, interiorYY = np.meshgrid(interiorX, interiorY)

    wavefuncs=np.array(wavefuncs)
    densities = np.abs(wavefuncs)**2
    interiorDensities = densities[:, interiorYIndices[:, None], interiorXIndices]

    os.makedirs("frames", exist_ok=True)

    i=0 
    maxDensity = np.max(interiorDensities)
    while i<len(tList):
        print(30*"\n", 50*"=")
        print(" Bose-Einstein Condensate Dynamics Simulator")
        print(" STEP 3 of 4 - Visualising Result")
        print("", 50*"=")
        print(" Post-Processing complete. Plotting results and constructing frames...")
        print(f" {i} out of {len(tList)} frames constructed ({(100 * i / len(tList)):.2f}% )")
        curDensities = interiorDensities[i]
        plt.figure(figsize=(8,5))
        img = plt.pcolor(interiorXX, interiorYY, curDensities, cmap="nipy_spectral", norm=LogNorm(vmin=1e-4, vmax=maxDensity))
        #img = plt.pcolor(interiorXX, interiorYY, curDensities, cmap="nipy_spectral", vmin=0, vmax=maxDensity)
        plt.title(f"2D Solution for GPE Model of Bose-Einstein Condensate, t={tList[i]:.3f} (Frame {i})")
        #plt.title(f"2D Solution for GPE Model of Bose-Einstein Condensate")
        plt.xlabel("x-position")
        plt.ylabel("y-position")
        plt.colorbar(img, label="Condensate Density")
        frame_path = f"frames/frame_{i:04d}.png"
        plt.savefig(frame_path)
        plt.close()
        i+=1 
    print(30*"\n")
    print(f" {len(tList)} out of {len(tList)} frames constructed ({100:.2f}% )")
    print(" Frames Constructed. Initialising next process...")

    j=0 
    output_path ="/Users/kanlachlan/Documents/VS_Code/Personal Projects/Bose-Einstein_Condensates/2D_FLAT_Sim_Output.mp4"
    with imageio.get_writer(output_path, fps=fpsCustom) as writer:
        while j<len(tList):
            print(30*"\n", 50*"=")
            print(" Bose-Einstein Condensate Dynamics Simulator")
            print(" STEP 4 of 4 - Finalising Result")
            print("", 50*"=")
            print(" Frames constructed. Assembling video...")
            print(f" {j} out of {len(tList)} frames assembled ({(100 * j / len(tList)):.2f}% )")
            frame_path = f"frames/frame_{j:04d}.png"
            image = imageio.imread(frame_path)
            writer.append_data(image)
            j+=1
    print(30*"\n", 50*"=")
    print(" Bose-Einstein Condensate Dynamics Simulator")
    print(" STEP 4 of 4 - Finalising Result")
    print("", 50*"=")
    print(" Frames constructed. Assembling video...")
    print(f" {len(tList)} out of {len(tList)} frames assembled (100 % )")
    print(" Video saved as 2D_Sim_Output.mp4 with path ", output_path)

def importGroundState(tauMax:float, ansatzFunc:callable, L:float,  dtau:float, dL:float,  gamma:float, m:float, customParam:float, plottingBorders:float, fpsCustom:float=100, customPotential:callable=potential):
    return initstateFinder.groundFinder(ansatzFunc, customPotential, gamma, m, customParam, L, dL, dtau, tauMax, 3, False )

def fullSolver(tmax:float, tauMax:float, ansatzFunc:np.ndarray, L:float,  dt:float, dtau:float,  dL:float,  gamma:float, m:float, customParam:float, plottingBorders:float, fpsCustom:float=100, customPotential:callable=potential, perterbation:callable=vortexPerturbation):
    initialPsi, diverge=initstateFinder.groundFinder(ansatzFunc, customPotential, gamma, m, customParam, L, dL, dtau, tauMax, 3, False )
    plotterFLAT2(tmax, initialPsi, L,  dt, dL,  gamma, m, customParam, plottingBorders, fpsCustom, customPotential, diverge, perterbation)



#fullSolver(6.2, 5, ansatz.vortexGaussian, 8, 0.012, 0.012,  0.012, -1, 1, 1/3, 3/2, 10, potentialFuncs.harmonicOscillator) 
#fullSolver(6.2, 10, ansatz.thomas_fermi, 10, 0.012, 0.012,  0.012, 100, 1, 1/3, 2, 10, potentialFuncs.harmonicOscillator, perturbations.vortexPerturbation) 
fullSolver(6.2, 10, ansatz.wideGaussian, 8, 0.012, 0.012,  0.012, -1, 1/5, 1/5, 3/2, 10, potentialFuncs.harmonicOscillator, perturbations.vortexPerturbation) 


#Saved runs is now outside of folder. There is a new folder called "Saved Results"



#REMEMBER TO SWITCH BACK TO LOGARITHMIC SCALING 

#For when using initstateFinder: MINIMUM: tauMax ~ 1/omega0, for safety and accuracy reccommend tauMax ~ 2 * 1/omega0
#DO NOT have more than 120 frames for and L/dL = 1725, to have more frames (by increasing tmax or decreasing dt), arrange the input parameters such that L/dL is smaller
#DO NOT have more than 502 frames for and L/dL = 800, to have more frames (by increasing tmax or decreasing dt), arrange the input parameters such that L/dL is smaller
#colormap: nipy_spectral
#Recommended fps between 10 and 100


#LEGACY COMMENTS AND INITIAL CONDITIONS (LEGACY == BEFORE GROUNDSTATE FINDER WAS MADE)
#FOR vortex1 ansatz initial condition: High mass => looks less discretised?

#plotter(6, Psi, 5, 0.01, 0.01, -1, 1/5, 2, 3/2)
#plotter(.72, Psi, 5, 0.01/2, 0.01/3, 100, 1/5, 2, 3/2+2, 10)  
#plotter(.72, Psi, 5, 0.01/2, 0.01/3, 100, 1/5, -100, 3/2, 10) 
#plotter(2, ansatz.vortex2, 5, 0.01*2, 0.01/3, -1, 1/5, 2, 1.2, 10, potentialFuncs.harmonicOscillator) 
#plotter(.72, ansatz.vortex1, 5, 0.01/2, 0.01/3, 100, 1/5, 1/2, 3/2, 10, potentialFuncs.harmonicOscillator) 
#plotterFLAT(.72, ansatz.vortex1, 5, 0.01/2, 0.01/3, 100, 1/5, 1/2, 3/2, 10, potentialFuncs.harmonicOscillator) 
#plotterFLAT(5, ansatz.vortex1, 8, 0.01, 0.01, 100, 100, 1/2, 3/2, 10, potentialFuncs.harmonicOscillator) 
#plotterFLAT(.72, Psi, 5, 0.01/2, 0.01/3, 100, 1/5, -100, 3/2, 10) 
#plotterFLAT(5, ansatz.vortex1, 8, 0.01, 0.01, 100, 10, 1/2, 3/2, 10, potentialFuncs.harmonicOscillator) 
#plotterFLAT(5, ansatz.vortex1, 8, 0.01, 0.01, -50, 5, 1/2, 3/2, 10, potentialFuncs.harmonicOscillator) 
#plotterFLAT(5.8, ansatz.vortex1, 8, 0.012, 0.012, -1/2, 10, 0.8, 3/2, 10, potentialFuncs.harmonicOscillator) 
#plotterFLAT(5.5, ansatz.vortex1, 8, 0.01, 0.012, 1.26, 50, 1/10, 3/2, 10, potentialFuncs.harmonicOscillator) 
#plotterFLAT(5, ansatz.normalised_vortex, 8, 0.01, 0.01, 30, 100, 1/2, 3/2, 10, potentialFuncs.harmonicOscillator) 

#plotterFLAT(5.5, ansatz.thomas_fermi, 8, 0.012, 0.01, 30, 80, 1/5, 3/2, 10, potentialFuncs.harmonicOscillator) 


#started solver 12:02
#ground state found 12:08
#perturbation dynamics solved 12:12
#post-processing complete 12:17
#frames created 12:24
#video assembled 12:24
#total time 22 mins