import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os 
import imageio.v2 as imageio

#hbar=c=1

m=1/5
omega0=1

def Psi(x,t):
    psi= 1/np.cosh(x-1) #np.exp(-15*(x-1/2)**2)+np.exp(-15*(x+1)**2)+np.exp(-15*(x-2)**2)+np.exp(-15*(x+2)**2)
    return psi

def potential(x):
    return(1/2)*m*omega0**(2)*x**2

def nonLinear(psi:float, gamma:float, potentialVal:float)->float:
    """
    Return the nonlinear component of the GP equation when psi=Psi(x,t)
    """
    return gamma*np.abs(psi)**2 + potentialVal

def nextStep(psi:np.ndarray, L:float, dt:float, dx:float, gamma:float)->float:
    """
    Return next step of the wavefunction psi=Psi(x,t+dt) , note that the use of fft implies periodic boundary conditions. To avoid effects from the periodic boundaries, 
    set L to be much greater than the range of motion of the wavefunction, ie. L>>x
    """
    x=np.linspace(-L,L, int(2*L/dx+1))
    potentials=potential(x)
    N=nonLinear(psi,gamma,potentials)
    k=2*np.pi*np.fft.fftfreq(len(x), d=dx)
    F=np.exp(-1j*dt*k**2/2)*np.fft.fft(np.exp(-1j*dt*N)*psi)
    return np.exp(-1j*dt*(N)/2)*np.fft.ifft(F)

def getPsiVals(Psi:callable, L:float, dx:float, t:float):
    x=np.linspace(-L,L,int(2*L/dx+1))
    psiList=[]
    for xval in x:
        psiList.append(Psi(xval,t))
    return np.array(psiList)


def loopy(initialPsi:callable, L:float, t0:float, tmax:float, dt:float, dx:float, gamma:float)->float:
    """
    Apply nextStep() repeatedly until tmax is reached
    """
    print("Solver running...")
    t=t0
    xList=np.linspace(-L,L,int(2*L/dx+1))  
    psi=getPsiVals(initialPsi,L,dx,t)
    tList=[t0]
    psiList=[psi/np.sqrt(np.sum(np.abs(psi)**2) * dx)]
    while t<=tmax:
        print(30*"\n","Solver running...")
        print(f" Current Simulation time: {t:.2f} || Maximum simulation time: {tmax:.2f} || Progress: {(100 * t / tmax):.2f}%")
        psi=nextStep(psi, L, dt, dx, gamma)
        t+=dt 
        psiList.append(psi/np.sqrt(np.sum(np.abs(psi)**2) * dx))
        tList.append(t)
    return tList,xList,psiList

def plotter(initialPsi:callable, L:float, t0:float, tmax:float, dt:float, dx:float, gamma:float, plottingBorders:float)->float:
    """
    Plot psi amplitude as a function of x and t
    """
    tList,xList,psiList=loopy(initialPsi, L, t0, tmax, dt, dx, gamma)
    interiorXIndices = np.where(np.abs(xList) <= plottingBorders)[0]
    interiorX=xList[interiorXIndices]
    T, X=np.meshgrid(tList,interiorX,indexing="ij")
    densities=np.abs(psiList)**2
    interiorDensities=densities[:,interiorXIndices]

    print("Solving complete. Plotting results...")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T, X, interiorDensities,cmap=cm.nipy_spectral)
    ax.set_xlabel("Time")
    ax.set_ylabel("Position")
    ax.set_zlabel("Density")
    plt.show()
    return T, X , densities

def videoMaker(initialPsi:callable, L:float, t0:float, tmax:float, dt:float, dx:float, gamma:float, plottingBorders:float)->float:
    tList,xList,psiList=loopy(initialPsi, L, t0, tmax, dt, dx, gamma)
    i=0 
    interiorXIndices = np.where(np.abs(xList) <= plottingBorders)[0]
    interiorX=xList[interiorXIndices]
    densities=np.abs(psiList)**2
    interiorDensities=densities[:,interiorXIndices]
    potentials=potential(interiorX)
    os.makedirs("frames", exist_ok=True)
    while i<len(tList):
        print(30*"\n","Solving complete. Plotting results and constructing frames...")
        print(f" {i} out of {len(tList)} frames constructed ({(100 * i / len(tList)):.2f}% )")
        plt.plot(interiorX, interiorDensities[i], color="black",label="|Psi^2|")
        plt.plot(interiorX, potentials, label="Heuristic Potential Plot")
        plt.title(f"Gross-Pitaevskii Model of Bose-Einstein Condensate, t={tList[i]:.2f} (Frame {i})")
        plt.xlabel("Position")
        plt.ylabel("Condensate Density")
        plt.legend(loc='upper right',  bbox_to_anchor=(1.129, 1))
        plt.ylim(-1/10, 5/2) #Set to (-1/10, 3/2) if not simulating anything with large amplitude
        frame_path = f"frames/frame_{i:04d}.png"
        plt.savefig(frame_path)
        plt.close()
        i+=1
    frames = []
    j=0
    while j<=len(tList):
        print(30*"\n", "Frames constructed. Assembling video...")
        print(f" {j} out of {len(tList)} frames assembled ({(100 * j / len(tList)):.2f}% )")
        frame_path = f"frames/frame_{j:04d}.png"
        image = imageio.imread(frame_path)
        frames.append(image)
        j+=1 
    output_path ="/Users/kanlachlan/Documents/VS_Code/Personal Projects/Bose-Einstein_Condensates/1D_Sim_Output.mp4"
    imageio.mimsave(output_path, frames, fps=100) #set to 100 fps if not simulating anything special
    print("Video saved as 1D_Sim_Output.mp4 with path ", output_path)

#plotter(Psi, 50, 0, 2, 0.001, 0.001, 1, 3/2)
videoMaker(Psi, 100, 0, 15, 0.01, 0.01, -3, 5/2) #gamma<0=> attract to force center, gamma>0 => repel from force center
