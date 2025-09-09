# Bose-Einstein Condensate Solver
This project predicts the evolution of the density for a sample of Bose-Einstein Condensate. The density is taken as $$|\psi|^2$$, where $$\psi=\psi(x,y,t)$$ is the wavefunction of the
system. To achieve this, the program solves the time-dependent Gross-Pitaevskii Equation (TDGPE), given by the following:

$$
i\frac{\partial\psi}{\partial t}=-\frac{1}{2m}\nabla^2\psi+(U+\gamma|\psi|^2)\psi
$$

Where $$\gamma$$ is a constant describing the nonlinear interaction within the condensate. $$\gamma<0$$ represents attractive behaviour and $$\gamma>0$$ represents repulsive behaviour. 
Solutions for the 1D and 2D cases are supported. Results are presented in the format of a video output, which shows the condensate density evolve over time. The condensate density is 
indicated either by a 3D graph or a colourmap. 

## Split Step Method
The main solver files `twoD_splitStep.py` and `OneD_splitStep.py` are numerical solvers for the 2D and 1D case of GPE respectively. They utilise a method called "Split-Step Strang Splitting", which 
splits the main equation into its dispersion (kinetic energy) and nonlinear (potential energy and interaction term) parts, treating them separately.

## Initial Conditions
There are two main types of initial conditions - Ansatze and perturbed ground states. Both will initialise a wavefunction $$\psi(x,y,0)$$ This section will provide an overview of them.

### Ansatze 
The ansatze are stored in the file `ansatz.py`, and can be used directly as an initial wavefunction for the solver. 
However, this is not very reliable as not 
all initial conditions are physically accurate. After downloading the files, you are free to add any other custom ansatze into the file for your desired initial wavefunction. 

### Perturbed Ground State 
This is a much more realistic method. This method starts with an ansatz, and runs it through `initstateFinder.py`, where imaginary time ($$it\to\tau$$) is propagated through the GPE in an attempt to converge to a ground state 
(There will be warnings if the result diverges). After the ground state has been found, a perturbation of your choice from `perturbations.py` will be appended to the ground state (again, after downloading the files, 
you are free to add custom perturbations to the file). At the time of writing, this file only contains vortices by default. The perturbed ground state will be used as the initial condition of the main solver.  

## Potentials
The main goal of this project is to study the behaviour of BECs under an optical trap, which has the potential of a harmonic oscillator. More complicated potentials are found in `potentialFuncs.py` and are available for customisation 
after download. 

## Boundary Conditions
Since this project relies on NumPy's Fast Fourier Transform (FFT) algorithm to compute the steps in the dispersion term, periodic boundary conditions are assumed. Please make sure all wavefunctions and potentials 
decay to 0 as they approach the user-defined simulation borders.

## Notes 
Notes and technical details on the physics of this project are available in the `BEC Notes` folder. A fully rendered PDF is also in the folder available for viewing.  

## Cautions and Dependencies
The output paths for the solutions are hardcoded to my computer. Remember to change them to your desired path before use. Also rememebr to install the following dependencies before running the code:
```
numpy
matplotlib
imageio
```
These are crucial fo numerical and visualisation methods. Without these installed, the code will not run. 
