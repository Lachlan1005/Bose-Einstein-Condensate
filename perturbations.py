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

def vortexPerturbation(x,y):
    l=1
    theta=np.arctan2(y,x)
    return np.exp(1j*l*theta)

def largeLvortexPerturbation(x,y):
    l=2
    theta=np.arctan2(y,x)
    return np.exp(1j*l*theta)